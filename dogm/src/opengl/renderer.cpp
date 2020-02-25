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
#include "opengl/renderer.h"

#define _USE_MATH_DEFINES
#include <math.h>

Renderer::Renderer(int grid_size, float fov, float grid_range, float max_range)
	: grid_size(grid_size)
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

	float range = 2.0f * (max_range / grid_range);

	std::vector<Vertex> vertices;
	generateCircleSegmentVertices(vertices, fov, 2.0f, 0.0f, -1.0f);

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
	vertices.push_back(Vertex(glm::vec2(cx, cy), glm::vec2(0.0f, 0.0f)));

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

		vertices.push_back(Vertex(glm::vec2(cx + x, cy + y),
			glm::vec2((angle - startAngle) / fov, 1.0f)));
	}
}
