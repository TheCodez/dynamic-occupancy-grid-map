// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#pragma once

#include <GL/glew.h>
#include <cuda_gl_interop.h>

class Texture
{
public:
    Texture(int width, int height, float anisotropy_level = 0.0f);
    ~Texture();

    void beginCudaAccess(cudaSurfaceObject_t* surfaceObject);
    void endCudaAccess(cudaSurfaceObject_t surfaceObject);

    void generateMipMap();

    void bind(GLuint tex_unit);

private:
    GLuint texture;
    GLuint unit;

    cudaGraphicsResource_t resource;
};
