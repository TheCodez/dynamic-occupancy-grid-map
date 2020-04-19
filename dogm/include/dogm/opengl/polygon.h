// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

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
