/**
  \ingroup NNLib
  \file    PytorchFrontend
  \brief   This file contains the implementation for class PytorchFrontend.
  \author  rodrigodzf@gmail.com
  \date    2022-03-28
*/

#pragma once

#include <string>
#include <vector>
#include <array>
#include <memory>

#include "BaseNN.h"
#include <torch/script.h>


class PytorchFrontend : public BaseNN {
private:
    torch::jit::script::Module mModule;
    std::vector <torch::jit::IValue> mInputs;
private:
    void printDebug();

public:
    PytorchFrontend();

    ~PytorchFrontend();

    bool load(const std::string &filename) override;

    // Sets the sample rate and buffer size of a Neutone model to allocate all the internal buffers
    bool setDawSampleRateAndBufferSize(const int sampleRate, const int bufferSize) {
        mModule.get_method("set_daw_sample_rate_and_buffer_size")({sampleRate, bufferSize});
        return true;
    }

    // Resets the state in a Neutone model
    bool reset() {
        std::vector <torch::jit::IValue> val{};
        mModule.get_method("reset")(val);
        return true;
    }

    // Perform inference on a buffer of audio and parameters at audio rate
    bool forward(const std::vector <torch::jit::IValue> &inputs) {
        auto tensorOut = mModule.get_method("forward")(inputs).toTensor();
        return true;
    }
};
