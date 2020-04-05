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

    Framebuffer* getFrameBuffer() const { return framebuffer; }

private:
    void generateCircleSegmentVertices(std::vector<Vertex>& vertices, float fov, float radius, float cx, float cy);

private:
    int grid_size;

    Polygon* polygon;
    Shader* shader;
    Framebuffer* framebuffer;

    GLFWwindow* window;
};
