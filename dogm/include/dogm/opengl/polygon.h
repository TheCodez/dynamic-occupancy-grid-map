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

#include <GL/glew.h>
#include <glm/glm.hpp>

class Vertex
{
public:
    Vertex(const glm::vec2& pos) { this->pos = pos; }

    Vertex(const glm::vec2& pos, const glm::vec2& tex_coord)
    {
        this->pos = pos;
        this->tex_coord = tex_coord;
    }

    glm::vec2 pos;
    glm::vec2 tex_coord;
};

class Polygon
{
public:
    Polygon(Vertex* vertices, size_t num_vertices);
    ~Polygon();

    void draw();

private:
    GLuint vao;
    GLuint vbo;

    unsigned int vertices_count;
};
