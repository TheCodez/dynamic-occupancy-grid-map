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

#include <memory>
#include <vector>

class Renderer
{
public:
    Renderer(int grid_size, float fov, float grid_range, float max_range);

    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) = delete;
    Renderer& operator=(Renderer&&) = delete;

    void renderToTexture(Texture& polar_texture);

    std::shared_ptr<Framebuffer> getFrameBuffer() const { return framebuffer; }

private:
    int grid_size;

    std::unique_ptr<Polygon> polygon;
    std::unique_ptr<Shader> shader;
    std::shared_ptr<Framebuffer> framebuffer;

    std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> window{nullptr, glfwDestroyWindow};
};
