// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#pragma once

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "framebuffer.h"
#include "polygon.h"
#include "shader.h"
#include "texture.h"

#include <vector>

class Renderer
{
public:
    Renderer(int grid_size, float fov, float grid_range, float max_range);
    ~Renderer();

    void renderToTexture(Texture& polar_texture);

    std::shared_ptr<Framebuffer> getFrameBuffer() const { return framebuffer; }

private:
    int grid_size;

    Polygon* polygon;
    Shader* shader;
    Framebuffer* framebuffer;

    GLFWwindow* window;
};
