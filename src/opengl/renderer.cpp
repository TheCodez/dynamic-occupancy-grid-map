#include "opengl/renderer.h"

#define _USE_MATH_DEFINES
#include <math.h>

Renderer::Renderer(int width, int height, float fov, int polar_height)
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_VISIBLE, GL_FALSE);

	window = glfwCreateWindow(width, height, "GPU Occupancy Grid Map", nullptr, nullptr);
	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;
	glewInit();

	std::vector<Vertex> vertices;
	float radius = 2.0f * (polar_height / static_cast<float>(height));
	generateCircleSegmentVertices(vertices, 70.0f, radius, 0.0f, -1.0f);

	polygon = new Polygon(vertices.data(), vertices.size());
	shader = new Shader();
}

Renderer::~Renderer()
{
	delete polygon;
	delete shader;

	glfwTerminate();
}

void Renderer::render(Texture& texture)
{
	glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	texture.bind(0);
	texture.generateMipMap();

	shader->use();
	polygon->draw();
}

void Renderer::generateCircleSegmentVertices(std::vector<Vertex>& vertices, float fov, float radius, float cx, float cy)
{
	vertices.push_back(Vertex(glm::vec2(cx, cy), glm::vec2(0.0, 0.0)));

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
			glm::vec2((angle - startAngle) / fov, 1)));
	}
}
