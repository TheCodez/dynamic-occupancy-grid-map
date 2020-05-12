// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "mapping/opengl/polygon.h"

#include <cstddef>

Polygon::Polygon(Vertex* vertices, size_t numVertices)
{
    vertices_count = numVertices;

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, numVertices * sizeof(Vertex), vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, tex_coord));

    glBindVertexArray(0);
}

Polygon::~Polygon()
{
    glDeleteVertexArrays(1, &vao);
}

void Polygon::draw()
{
    glBindVertexArray(vao);

    glDrawArrays(GL_TRIANGLE_FAN, 0, vertices_count);

    glBindVertexArray(0);
}
