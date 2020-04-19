// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#pragma once

#include <GL/glew.h>
#include <cuda_gl_interop.h>

class Framebuffer
{
public:
    Framebuffer(int width, int height);
    ~Framebuffer();

    void beginCudaAccess(cudaSurfaceObject_t* surfaceObject);
    void endCudaAccess(cudaSurfaceObject_t surfaceObject);

    void bind();
    void unbind();

private:
    GLuint framebuffer;
    GLuint texture;

    cudaGraphicsResource_t resource;
};
