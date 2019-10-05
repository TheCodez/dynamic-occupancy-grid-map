#pragma once

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "polygon.h"
#include "shader.h"
#include "texture.h"
#include "framebuffer.h"

#include <vector>

class Renderer
{
public:
	Renderer(int width, int height, float fov);
	~Renderer();

	void render(Texture& texture);

	Framebuffer* getFrameBuffer() const { return framebuffer; }

private:
	void generateCircleSegmentVertices(std::vector<Vertex>& vertices, float fov, float radius, float cx, float cy);

private:
	Polygon* polygon;
	Shader* shader;
	Framebuffer* framebuffer;

	GLFWwindow* window;
};

