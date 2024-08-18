# Ray Tracer in C++

This project implements a basic ray tracer in C++ that supports the rendering of scenes with spheres, triangles, and meshes. The ray tracer computes intersections of camera rays with geometric primitives, handles shadows, and applies lighting to generate the final image.

## Features

- **Mesh Parsing**: The code can parse mesh data from a file and store it in a vector of tuples.
- **Sphere Intersection**: Computes intersections between rays and spheres, taking into account the sphere's material and center.
- **Triangle Intersection**: Computes intersections between rays and triangles, using the Möller-Trumbore intersection algorithm.
- **Mesh Intersection**: Handles intersections between rays and arbitrary meshes composed of triangles.
- **Shadow Computation**: Determines whether a point is in shadow by tracing shadow rays from the intersection point to the light sources.
- **Lighting Calculation**: Computes the final color of a point using ambient, diffuse, and specular lighting models.

## File Structure

- **`parseMeshes`**: Reads mesh data from a file and stores it in a vector of tuples.
- **`computeSphereIntersection`**: Calculates the intersection of a ray with a sphere.
- **`intersectRayTriangle`**: Determines if a ray intersects a triangle using the Möller-Trumbore algorithm.
- **`computeTriangleIntersection`**: Computes the intersection of a ray with a triangle and retrieves the corresponding material.
- **`computeMeshIntersection`**: Calculates the intersection of a ray with a mesh.
- **`computeCameraRay`**: Casts a ray from the camera and determines the closest intersection with any object in the scene.
- **`computeShadow`**: Checks if an intersection point is in shadow by tracing a shadow ray towards the light source.
- **`computeLighting`**: Computes the lighting at the intersection point using the Phong reflection model.

## Dependencies

- **C++ Standard Library**: The code uses standard C++ libraries such as `<vector>`, `<tuple>`, `<cmath>`, and `<limits>`.
- **Additional Libraries**: None required, the code is self-contained.

## How to Run

1. **Compile** the C++ code using a C++11 or higher compatible compiler:
   ```sh
   g++ -std=c++11 raytracer.cpp -o raytracer
   ```

2. **Execute** the compiled program:
   ```sh
   ./raytracer
   ```

3. The program will read mesh data from a file, compute intersections, and generate an image based on the specified camera, lights, and materials.

## Input Files

- **Mesh File**: The code expects a mesh file that contains mesh definitions. Each mesh is defined by a series of vertex indices and material properties.

## Example Usage

An example of how to use the code to parse a mesh file and compute intersections is provided in the main function. Customize the scene by modifying the mesh file, camera settings, and lighting configuration.
