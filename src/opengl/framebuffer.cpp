#include "opengl/framebuffer.h"
#include "cuda_utils.h"

#include <memory.h>
#include <cstdio>

Framebuffer::Framebuffer(int width, int height)
{
	glGenFramebuffers(1, &framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

	CHECK_ERROR(cudaGraphicsGLRegisterImage(&resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

Framebuffer::~Framebuffer()
{
	glDeleteFramebuffers(1, &framebuffer);
	glDeleteTextures(1, &texture);
}

void Framebuffer::beginCudaAccess(cudaSurfaceObject_t* surfaceObject)
{
	CHECK_ERROR(cudaGraphicsMapResources(1, &resource, 0));

	cudaArray_t cudaArray;
	CHECK_ERROR(cudaGraphicsSubResourceGetMappedArray(&cudaArray, resource, 0, 0));

	cudaResourceDesc resourceDesc;
	memset(&resourceDesc, 0, sizeof(cudaResourceDesc));
	resourceDesc.resType = cudaResourceTypeArray;
	resourceDesc.res.array.array = cudaArray;

	CHECK_ERROR(cudaCreateSurfaceObject(surfaceObject, &resourceDesc));
}

void Framebuffer::endCudaAccess(cudaSurfaceObject_t surfaceObject)
{
	CHECK_ERROR(cudaGraphicsUnmapResources(1, &resource, 0));
	CHECK_ERROR(cudaDestroySurfaceObject(surfaceObject));
}

void Framebuffer::bind()
{
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
}

void Framebuffer::unbind()
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
