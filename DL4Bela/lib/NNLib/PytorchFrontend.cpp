#include "PytorchFrontend.h"

#include "Log.h"

PytorchFrontend::PytorchFrontend() {}

PytorchFrontend::~PytorchFrontend() {}

void PytorchFrontend::printDebug() {}

bool PytorchFrontend::load(const std::string &filename) {
    // Disable for now for consistent baseline benchmarks
    torch::jit::setGraphExecutorOptimize(false);
    torch::jit::getProfilingMode() = false;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        mModule = torch::jit::load(filename);
        // This is useful for larger CNN models, so keep disabled for now
        // mModule = torch::jit::optimize_for_inference(mModule);
        mModule.eval();
    }
    catch (const std::exception &e) {
        NN_LOG(ERROR) << "error loading the model" << e.what();
        std::exit(1);
    }
    return true;
}
