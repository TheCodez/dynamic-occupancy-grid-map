// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "mapping/opengl/shader.h"
#include <iostream>

const char* vertex_source = R"glsl(
    #version 150

    in vec2 position;
	in vec2 texCoord;	
	out vec2 texCoord0;

    void main()
    {
        gl_Position = vec4(position, 0.0, 1.0);
		texCoord0 = texCoord;
    }
)glsl";

const char* fragment_source = R"glsl(
    #version 150

	in vec2 texCoord0;
	out vec4 fragColor;
	
	uniform sampler2D tex;

    void main()
    {
		vec2 uv = vec2(1.0 - (texCoord0.s / (texCoord0.t + 1e-10)), texCoord0.t);
		fragColor = texture(tex, uv);
    }
)glsl";

Shader::Shader()
{
    program = glCreateProgram();

    vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_source, nullptr);
    glCompileShader(vertex_shader);
    checkShaderError(vertex_shader);

    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_source, nullptr);
    glCompileShader(fragment_shader);
    checkShaderError(fragment_shader);

    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);

    glBindAttribLocation(program, 0, "position");
    glBindAttribLocation(program, 1, "texCoord");

    glLinkProgram(program);
}

Shader::~Shader()
{
    glDeleteProgram(program);
    glDeleteShader(fragment_shader);
    glDeleteShader(vertex_shader);
}

void Shader::checkShaderError(GLuint shader)
{
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

    if (status == GL_FALSE)
    {
        char buffer[512];
        glGetShaderInfoLog(shader, 512, nullptr, buffer);
        std::cerr << buffer << "'" << std::endl;
    }
}

void Shader::use()
{
    glUseProgram(program);
}
