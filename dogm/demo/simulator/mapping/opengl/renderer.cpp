// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "mapping/opengl/renderer.h"

#include <cmath>

Renderer::Renderer(int grid_size, float fov, float grid_range, float max_range) : grid_size(grid_size)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GL_FALSE);

    window = glfwCreateWindow(grid_size, grid_size, "GPU Occupancy Grid Map", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    glewInit();

    std::vector<Vertex> vertices;

    // center vehicle in the middle
    float range = 2.0f * (max_range / grid_range);
    generateCircleSegmentVertices(vertices, fov, range, 0.0f, 0.0f);
    // generateCircleSegmentVertices(vertices, fov, range, 0.0f, -1.0f);

    polygon = new Polygon(vertices.data(), vertices.size());
    shader = new Shader();
    framebuffer = new Framebuffer(grid_size, grid_size);
}

Renderer::~Renderer()
{
    delete polygon;
    delete shader;
    delete framebuffer;

    glfwTerminate();
}

void Renderer::renderToTexture(Texture& polar_texture)
{
    glViewport(0, 0, grid_size, grid_size);
    framebuffer->bind();

    // red=occ, green=free
    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_CULL_FACE);

    polar_texture.bind(0);
    polar_texture.generateMipMap();

    shader->use();
    polygon->draw();

    framebuffer->unbind();
}

void Renderer::generateCircleSegmentVertices(std::vector<Vertex>& vertices, float fov, float radius, float cx, float cy)
{
    vertices.emplace_back(Vertex(glm::vec2(cx, cy), glm::vec2(0.0f, 0.0f)));

    // fix 1 off error
    fov += 1.0f;

    float halfFov = fov / 2;
    float startAngle = 90 - halfFov;
    float endAngle = 90 + halfFov;

    for (int angle = startAngle; angle <= endAngle; angle++)
    {
        float angle_radians = angle * M_PI / 180.0f;

        float x_val = cos(angle_radians);
        float y_val = sin(angle_radians);

        float x = radius * x_val;
        float y = radius * y_val;

        vertices.emplace_back(Vertex(glm::vec2(cx + x, cy + y), glm::vec2((angle - startAngle) / fov, 1.0f)));
    }
}
