// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#pragma once

#include <GL/glew.h>

class Shader
{
public:
    Shader();
    ~Shader();

    void use();

private:
    void checkShaderError(GLuint shader);

private:
    GLuint program;
    GLuint vertex_shader;
    GLuint fragment_shader;
};
