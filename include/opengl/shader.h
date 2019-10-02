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

