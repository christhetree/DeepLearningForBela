#include <algorithm>
#include <numeric>
#include <fstream>
#include <memory>
#include <iomanip>
#include <chrono>

#include "Utils.h"
#include "argparse.h"
#include "Log.h"

#include "PytorchFrontend.h"


// Calculates the standard deviation of a list of doubles
double calculateStandardDeviation(const std::vector<double> &values) {
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double sumSquaredDiff = 0.0;

    for (const auto &value: values) {
        double diff = value - mean;
        sumSquaredDiff += diff * diff;
    }

    double variance = sumSquaredDiff / values.size();
    return std::sqrt(variance);
}

// Benchmarks a Neutone model
int main(int argc, char *argv[]) {
    argparse::ArgumentParser program("Benchmark for Neutone models.");

    // Path to model.nm file
    program.add_argument("-m", "--model").required().help("Model name.");

    // DAW buffer size
    program.add_argument("-b", "--buffer_size")
            .required()
            .scan<'i', int>();

    // DAW sample rate
    program.add_argument("-s", "--sample_rate")
            .default_value(44100)
            .scan<'i', int>();

    // DAW no. of channels (1 or 2)
    program.add_argument("-c", "--n_channels")
            .default_value(2)
            .scan<'i', int>();

    // No. of warmup iterations
    program.add_argument("-w", "--warmup_iterations")
            .default_value(20)
            .scan<'i', int>();

    // No. of benchmark iterations
    program.add_argument("-n", "--average_iterations")
            .default_value(200)
            .scan<'i', int>();

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err) {
        NN_LOG(ERROR) << err.what();
        NN_LOG(ERROR) << program;
        std::exit(1);
    }

    auto modelName = program.get<std::string>("-m");
    int bufferSize = program.get<int>("buffer_size");
    int sampleRate = program.get<int>("sample_rate");
    int nChannels = program.get<int>("n_channels");
    // Neutone models have 4 parameters by default
    const int nParams = 4;

    unsigned int iterations = program.get<int>("average_iterations");
    unsigned int warmupIterations = program.get<int>("warmup_iterations");

    NN_LOG(INFO) << "Model name " << modelName;
    NN_LOG(INFO) << "Buffer size " << bufferSize;

    NN_LOG(INFO) << "Creating Pytorch pipeline";
    PytorchFrontend nn;

    NN_LOG(INFO) << "Loading model";
    nn.load(modelName);
    NN_LOG(INFO) << "Successfully loaded model";

    // Create an audio buffer of random noise
    auto inAudio = torch::rand({nChannels, bufferSize});
    // Create a parameter values of random noise
    auto inParams = torch::rand({nParams, bufferSize});
    // Create the input for Neutone models
    std::vector <torch::jit::IValue> inputs;
    inputs.push_back(inAudio);
    inputs.push_back(inParams);
    NN_LOG(INFO) << "Successfully made inputs";

    // Set the DAW sample rate and buffer size for the Neutone model
    nn.setDawSampleRateAndBufferSize(sampleRate, bufferSize);
    NN_LOG(INFO) << "Successfully set sample rate to " << sampleRate;
    NN_LOG(INFO) << "Successfully set buffer size to " << bufferSize;

    // Vector for storing duration values for each warmup iteration
    std::vector<double> warmupDurations;

    NN_LOG(INFO) << "Starting warmup iterations";
    for (unsigned int i = 0; i < warmupIterations; i++) {
        const auto start_time = std::chrono::high_resolution_clock::now();
        nn.forward(inputs);
        const auto duration = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - start_time);
        // Record inference duration in ms
        warmupDurations.push_back(duration.count());
    }
    float warmupAverage = -1.0;
    double warmupStd = -1.0;
    double warmupMin = -1.0;
    double warmupMax = -1.0;
    double warmupRTF = -1.0;
    if (warmupIterations > 0) {
        // Calculate statistics
        warmupAverage =
                std::accumulate(std::begin(warmupDurations), std::end(warmupDurations), 0.0) / warmupDurations.size();
        warmupStd = calculateStandardDeviation(warmupDurations);
        warmupMin = *std::min_element(warmupDurations.begin(), warmupDurations.end());
        warmupMax = *std::max_element(warmupDurations.begin(), warmupDurations.end());
        warmupRTF = (((double) bufferSize / (double) sampleRate) * 1000.0) / warmupAverage;
        NN_LOG(INFO) << "Warmup done, on average after " << warmupIterations << " iterations :" << std::fixed
                     << warmupAverage << "ms";
    }

    // Reset the Neutone model after warming up to clear internal state
    nn.reset();
    NN_LOG(INFO) << "Successfully reset model";

    // Vector for storing duration values for each benchmark iteration
    std::vector<double> durations;

    NN_LOG(INFO) << "Starting benchmark iterations";
    for (unsigned int i = 0; i < iterations; i++) {
        const auto start_time = std::chrono::high_resolution_clock::now();
        nn.forward(inputs);
        const auto duration = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - start_time);
        // Record inference duration in ms
        durations.push_back(duration.count());
    }

    // Calculate statistics
    float iterAverage = std::accumulate(std::begin(durations), std::end(durations), 0.0) / durations.size();
    double iterStd = calculateStandardDeviation(durations);
    double iterMin = *std::min_element(durations.begin(), durations.end());
    double iterMax = *std::max_element(durations.begin(), durations.end());
    double iterRTF = (((double) bufferSize / (double) sampleRate) * 1000.0) / iterAverage;
    NN_LOG(INFO) << "Inference done, on average after " << iterations << " iterations :" << std::fixed << iterAverage
                 << "ms";

    // Print out statistics for analysis
    std::cout << "============================================================\n";
    std::cout << sampleRate << "\n";
    std::cout << bufferSize << "\n";
    std::cout << nChannels << "\n";
    std::cout << "warmup\n";
    std::cout << warmupIterations << "\n";
    std::cout << warmupAverage << "\n";
    std::cout << warmupStd << "\n";
    std::cout << warmupMin << "\n";
    std::cout << warmupMax << "\n";
    std::cout << warmupRTF << "\n";
    std::cout << "iterations\n";
    std::cout << iterations << "\n";
    std::cout << iterAverage << "\n";
    std::cout << iterStd << "\n";
    std::cout << iterMin << "\n";
    std::cout << iterMax << "\n";
    std::cout << iterRTF << "\n";
}
