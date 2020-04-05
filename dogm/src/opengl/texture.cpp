/*
MIT License

Copyright (c) 2019 Michael Kösel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "dogm/opengl/texture.h"
#include "dogm/cuda_utils.h"

#include <cstdio>
#include <memory.h>
#include <stdio.h>

Texture::Texture(int width, int height, float anisotropy_level)
{
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width, height, 0, GL_RG, GL_FLOAT, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    GLint swizzleMask[] = {GL_RED, GL_GREEN, GL_ONE, GL_ONE};
    glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask);

    if (anisotropy_level > 0.0f)
    {
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy_level);
    }

    float color[] = {0.0f, 0.0f, 1.0f, 1.0f};
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, color);

    CHECK_ERROR(
        cudaGraphicsGLRegisterImage(&resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

    glBindTexture(GL_TEXTURE_2D, 0);
}

Texture::~Texture() { glDeleteTextures(1, &texture); }

void Texture::beginCudaAccess(cudaSurfaceObject_t* surfaceObject)
{
    CHECK_ERROR(cudaGraphicsMapResources(1, &resource, 0));

    cudaArray_t cudaArray;
    CHECK_ERROR(cudaGraphicsSubResourceGetMappedArray(&cudaArray, resource, 0, 0));

    cudaResourceDesc resourceDesc;
    memset(&resourceDesc, 0, sizeof(cudaResourceDesc));
    resourceDesc.resType = cudaResourceTypeArray;
    resourceDesc.res.array.array = cudaArray;

    CHECK_ERROR(cudaCreateSurfaceObject(surfaceObject, &resourceDesc));
}

void Texture::endCudaAccess(cudaSurfaceObject_t surfaceObject)
{
    CHECK_ERROR(cudaGraphicsUnmapResources(1, &resource, 0));
    CHECK_ERROR(cudaDestroySurfaceObject(surfaceObject));
}

void Texture::generateMipMap()
{
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);
}

void Texture::bind(GLuint texUnit)
{
    unit = texUnit;
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, texture);
}
