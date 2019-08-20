#include "OccupancyGridMap.h"

int main(int argc, const char** argv) 
{
	GridParams params;
	params.width = 120;
	params.height = 120;
	params.resolution = 0.1f;
	params.v = 2 * 10e6;
	params.vB = 2 * 10e5;
	params.pS = 0.99f;
	params.processNoisePosition = 0.02f;
	params.processNoiseVelocity = 0.8f;
	params.pB = 0.02f;

	OccupancyGridMap gridMap(params);
}