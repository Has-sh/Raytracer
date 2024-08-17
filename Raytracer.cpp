#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <tuple>
#include <cmath>
#include <limits>

class Color {
public:
    float r, g, b;
};

class Material {
public:
    int index;
    Color ambient;
    Color diffuse;
    Color specular;
    float specularExponent;
    Color mirror;
};

class PointLight {
public:
    int index;
    float x, y, z;
    Color intensity;
};

class Vertex {
public:
    float x, y, z;
};

class Camera {
public:
    float x, y, z;
    float gazeX, gazeY, gazeZ;
    float upX, upY, upZ;
    float left, right, bottom, top;
    float distance;
    int width, height;
};

void parseTag(const std::string& fileName, const std::string& tag, std::vector<std::string>& values) {
    std::ifstream inputFile(fileName);
    std::string line;
    while (std::getline(inputFile, line)) {
        if (line.find(tag) != std::string::npos) {
            while (std::getline(inputFile, line) && !line.empty()) {
                values.push_back(line);
            }
            break;
        }
    }
    inputFile.close();
}

Color parseBackgroundColor(const std::string& fileName) {
    std::vector<std::string> values;
    parseTag(fileName, "#BackgroundColor", values);
    Color color;
    if (values.size() == 1) {
        std::istringstream iss(values[0]);
        iss >> color.r >> color.g >> color.b;
    }
    return color;
}

int parseMaxRecursionDepth(const std::string& fileName) {
    std::vector<std::string> values;
    parseTag(fileName, "#MaxRecursionDepth", values);
    int depth = 0;
    if (values.size() == 1) {
        std::istringstream iss(values[0]);
        iss >> depth;
    }
    return depth;
}

float parseShadowRayEpsilon(const std::string& fileName) {
    std::vector<std::string> values;
    parseTag(fileName, "#ShadowRayEpsilon", values);
    float epsilon = 0.0f;
    if (values.size() == 1) {
        std::istringstream iss(values[0]);
        iss >> epsilon;
    }
    return epsilon;
}

Camera parseCamera(const std::string& fileName) {
    std::vector<std::string> values;
    parseTag(fileName, "#Camera", values);
    Camera camera;
    if (values.size() == 6) {
        std::istringstream issPos(values[0]);
        issPos >> camera.x >> camera.y >> camera.z;

        std::istringstream issGaze(values[1]);
        issGaze >> camera.gazeX >> camera.gazeY >> camera.gazeZ;

        std::istringstream issUp(values[2]);
        issUp >> camera.upX >> camera.upY >> camera.upZ;

        std::istringstream issNearPlane(values[3]);
        issNearPlane >> camera.left >> camera.right >> camera.bottom >> camera.top;

        std::istringstream issDist(values[4]);
        issDist >> camera.distance;

        std::istringstream issRes(values[5]);
        issRes >> camera.width >> camera.height;
    }
    return camera;
}

std::vector<Material> parseMaterials(const std::string& fileName) {
    std::vector<Material> materials;
    std::ifstream inputFile(fileName);
    std::string line;
    while (std::getline(inputFile, line)) {
        if (line == "#Material") {
            Material material;
            std::vector<std::string> values;
            while (std::getline(inputFile, line) && !line.empty()) {
                values.push_back(line);
            }
            if (values.size() == 6) {
                std::istringstream iss(values[0]);
                iss >> material.index;

                std::istringstream issAmbient(values[1]);
                issAmbient >> material.ambient.r >> material.ambient.g >> material.ambient.b;

                std::istringstream issDiffuse(values[2]);
                issDiffuse >> material.diffuse.r >> material.diffuse.g >> material.diffuse.b;

                std::istringstream issSpecular(values[3]);
                issSpecular >> material.specular.r >> material.specular.g >> material.specular.b;

                std::istringstream issSpecularExp(values[4]);
                issSpecularExp >> material.specularExponent;

                std::istringstream issMirror(values[5]);
                issMirror >> material.mirror.r >> material.mirror.g >> material.mirror.b;

                materials.push_back(material);
            }
        }
    }
    inputFile.close();
    return materials;
}

Color parseAmbientLight(const std::string& fileName) {
    std::vector<std::string> values;
    parseTag(fileName, "#AmbientLight", values);
    Color ambientLight;
    if (values.size() == 1) {
        std::istringstream iss(values[0]);
        iss >> ambientLight.r >> ambientLight.g >> ambientLight.b;
    }
    return ambientLight;
}

std::vector<PointLight> parsePointLights(const std::string& fileName) {
    std::vector<PointLight> pointLights;
    std::ifstream inputFile(fileName);
    std::string line;
    while (std::getline(inputFile, line)) {
        if (line == "#PointLight") {
            PointLight pointLight;
            std::vector<std::string> values;
            while (std::getline(inputFile, line) && !line.empty()) {
                values.push_back(line);
            }
            if (values.size() == 3) {
                std::istringstream issIndex(values[0]);
                issIndex >> pointLight.index;

                std::istringstream issPosition(values[1]);
                issPosition >> pointLight.x >> pointLight.y >> pointLight.z;

                std::istringstream issIntensity(values[2]);
                issIntensity >> pointLight.intensity.r >> pointLight.intensity.g >> pointLight.intensity.b;

                pointLights.push_back(pointLight);
            }
        }
    }
    inputFile.close();
    return pointLights;
}

std::vector<Vertex> parseVertexList(const std::string& fileName) {
    std::vector<Vertex> vertexList;
    std::vector<std::string> values;
    parseTag(fileName, "#VertexList", values);
    for (const auto& line : values) {
        std::istringstream iss(line);
        Vertex vertex;
        iss >> vertex.x >> vertex.y >> vertex.z;
        vertexList.push_back(vertex);
    }
    return vertexList;
}

std::vector<std::tuple<int, int, int, float>> parseSpheres(const std::string& fileName) {
    std::vector<std::tuple<int, int, int, float>> spheres;
    std::ifstream inputFile(fileName);
    std::string line;
    while (std::getline(inputFile, line)) {
        if (line == "#Sphere") {
            std::tuple<int, int, int, float> sphere;
            if (std::getline(inputFile, line)) {
                std::istringstream iss(line);
                iss >> std::get<0>(sphere);
            }
            if (std::getline(inputFile, line)) {
                std::istringstream iss(line);
                iss >> std::get<1>(sphere); 
            }
            if (std::getline(inputFile, line)) {
                std::istringstream iss(line);
                iss >> std::get<2>(sphere);
            }
            if (std::getline(inputFile, line)) {
                std::istringstream iss(line);
                iss >> std::get<3>(sphere); 
            }
            spheres.push_back(sphere);
        }
    }
    inputFile.close();
    return spheres;
}

std::vector<std::tuple<int, int,int, int, int>> parseTriangles(const std::string& fileName) {
    std::vector<std::tuple<int,int, int, int, int>> triangles;
    std::ifstream inputFile(fileName);
    std::string line;
    while (std::getline(inputFile, line)) {
        if (line == "#Triangle") {
            std::tuple<int, int,int, int, int> triangle;
            if (std::getline(inputFile, line)) {
                std::istringstream iss(line);
                iss >> std::get<0>(triangle);
            }
            if (std::getline(inputFile, line)) {
                std::istringstream iss(line);
                iss >> std::get<1>(triangle); 
            }
            if (std::getline(inputFile, line)) {
                std::istringstream iss(line);
                iss >> std::get<2>(triangle) >> std::get<3>(triangle)  >> std::get<4>(triangle); 
            }
            triangles.push_back(triangle);
        }
    }
    inputFile.close();
    return triangles;
}

std::vector<std::tuple<int, int, std::vector<int>>> parseMeshes(const std::string& fileName) {
    std::vector<std::tuple<int, int, std::vector<int>>> meshes;
    std::ifstream inputFile(fileName);
    std::string line;
    while (std::getline(inputFile, line)) {
        if (line == "#Mesh") {
            std::tuple<int, int, std::vector<int>> mesh;
            if (std::getline(inputFile, line)) {
                std::istringstream iss(line);
                iss >> std::get<0>(mesh); 
            }
            if (std::getline(inputFile, line)) {
                std::istringstream iss(line);
                iss >> std::get<1>(mesh); 
            }
            std::vector<int> indices;
            while (std::getline(inputFile, line)) {
                if (line.empty() || line == "#") 
                    break;
                std::istringstream iss(line);
                int index;
                while (iss >> index) {
                    indices.push_back(index);
                }
            }
            std::get<2>(mesh) = indices;
            meshes.push_back(mesh);
        }
    }
    inputFile.close();
    return meshes;
}

bool computeSphereIntersection(std::vector<Vertex> vertexList,std::vector<std::tuple<int, int, int, float>> spheres,std::vector<Material> materials , float cameraX,float cameraY,float cameraZ, float sx, float sy, float sz, float* t,  Material& sphereMaterial, Vertex& center) {
    int sphereVertexId;
    float radius, t1= std::numeric_limits<float>::infinity(), t2= std::numeric_limits<float>::infinity();
    bool intersectionFound = false;
    
    for (const auto& sphere : spheres) {

        sphereVertexId = std::get<2>(sphere);
        radius = std::get<3>(sphere);
        int index = std::get<1>(sphere);

        for (const auto& material : materials) {
            if(material.index==index){
                sphereMaterial=material;
                break;
            }
        }
        // Retrieve sphere center vertex from vertex list
        Vertex sphereCenter = vertexList[sphereVertexId - 1];
        // *rad=radius;
        center=sphereCenter;
        // Calculate components of the equation
        float ox = cameraX;
        float oy = cameraY;
        float oz = cameraZ;
        
        float dx = sx;
        float dy = sy;
        float dz = sz;

        float cx = sphereCenter.x;
        float cy = sphereCenter.y;
        float cz = sphereCenter.z;

        // Calculate t using the provided formula
        float a = dx * dx + dy * dy + dz * dz;
        float b = dx * (ox - cx) + dy * (oy - cy) + dz * (oz - cz);
        float c = ((ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz)) - (radius * radius);
        
        float discriminant = b * b - (a * c);
        
        if (discriminant >= 0) {

            t1 = (-b + std::sqrt(discriminant)) / (a);
            t2 = (-b - std::sqrt(discriminant)) / (a);
            
            if (t1 > 0 && t1 < t2) {
                intersectionFound = true;
                *t = t1;
            }
            else if (t2 > 0 && t2 < std::numeric_limits<float>::infinity()) {
                intersectionFound = true;
                *t = t2;
            }
        }
    }
    return intersectionFound;
}

bool intersectRayTriangle(const Vertex& a, const Vertex& b, const Vertex& c, float ox, float oy, float oz, float dx, float dy, float dz, float* t) {
    // Calculate edge vectors
    float ax = b.x - a.x;
    float ay = b.y - a.y;
    float az = b.z - a.z;

    float bx = c.x - a.x;
    float by = c.y - a.y;
    float bz = c.z - a.z;

    // Calculate determinant and check if ray and triangle are parallel
    float px = dy * bz - dz * by;
    float py = dz * bx - dx * bz;
    float pz = dx * by - dy * bx;
    float det = ax * px + ay * py + az * pz;

    if (det == 0.0f)
        return false;

    float invDet = 1.0f / det;

    // Calculate distance from v0 to ray origin
    float tx = ox - a.x;
    float ty = oy - a.y;
    float tz = oz - a.z;

    // Calculate u parameter and test bounds
    float u = (tx * px + ty * py + tz * pz) * invDet;
    if (u < 0.0f || u > 1.0f)
        return false;

    // Calculate qvec and v parameter
    float qx = ty * az - tz * ay;
    float qy = tz * ax - tx * az;
    float qz = tx * ay - ty * ax;
    float v = (dx * qx + dy * qy + dz * qz) * invDet;

    if (v < 0.0f || u + v > 1.0f)
        return false;

    // Calculate t, the distance from the ray origin to the intersection point
    *t = (bx * qx + by * qy + bz * qz) * invDet;

    return *t >= 0.0f;
}

bool computeTriangleIntersection(std::vector<Vertex> vertexList,std::vector<std::tuple<int, int, int, int, int>> triangles,std::vector<Material> materials, float cameraX, float cameraY, float cameraZ, float sx, float sy, float sz, float* t, Material& triangleMaterial, Vertex& v0, Vertex& v1, Vertex& v2) {
    bool intersectionFound = false;

    for (const auto& triangle : triangles) {

        int index = std::get<1>(triangle);
        for (const auto& material : materials) {
            if(material.index==index){
                triangleMaterial=material;
                break;
            }
        }

        int v0Index = std::get<2>(triangle);
        int v1Index = std::get<3>(triangle);
        int v2Index = std::get<4>(triangle);

        // Retrieve triangle vertices from vertex list
        v0 = vertexList[v0Index - 1];
        v1 = vertexList[v1Index - 1];
        v2 = vertexList[v2Index - 1];

        if (intersectRayTriangle(v0, v1, v2, cameraX,cameraY,cameraZ, sx,sy,sz,t)) {
            intersectionFound = true;
            return intersectionFound;   
        }
    }
    return intersectionFound;
}

bool computeMeshIntersection(std::vector<Vertex> vertexList, std::vector<std::tuple<int, int, std::vector<int>>> meshes ,std::vector<Material> materials, float cameraX, float cameraY, float cameraZ, float sx, float sy, float sz, float* t, Material& meshMaterial, Vertex& v0, Vertex& v1, Vertex& v2) {
    bool intersectionFound = false;
    
    for (const auto& mesh : meshes) {
        
        int index = std::get<1>(mesh);
        for (const auto& material : materials) {
            if(material.index==index){
                meshMaterial=material;
                break;
            }
        }

        const auto& indicesList = std::get<2>(mesh);
        for (int i = 0; i < indicesList.size(); i += 3) {

            int v0Index = indicesList[i];
            int v1Index = indicesList[i + 1];
            int v2Index = indicesList[i + 2];

            // Retrieve triangle vertices from vertex list
            v0 = vertexList[v0Index - 1];
            v1 = vertexList[v1Index - 1];
            v2 = vertexList[v2Index - 1];

            // Check for intersection with the triangle
            if (intersectRayTriangle(v0, v1, v2, cameraX,cameraY,cameraZ, sx,sy,sz,t)) {
                intersectionFound = true;
                return intersectionFound;
            }
        }
    }
    return intersectionFound;
}

std::tuple<std::vector<float>, Material> computeCameraRay(Camera camera, Color backgroundColor,std::vector<Vertex> vertexList,std::vector<std::tuple<int, int, int, float>> spheres,std::vector<std::tuple<int, int, int, int, int>> triangles, std::vector<std::tuple<int, int, std::vector<int>>> meshes,std::vector<Material> materials, int i, int j, float *sx, float *sy, float *sz, Vertex& normal) {
    std::vector<float> ray(3, 0.0f);
    Material material1,material2,material3,material;
    Vertex center, v0, v1, v2;
    int min_index = -1;
    float t1 = std::numeric_limits<float>::infinity();
    float t2 = std::numeric_limits<float>::infinity();
    float t3 = std::numeric_limits<float>::infinity();
    float min_t = std::numeric_limits<float>::infinity();

    // Compute m = e + -wd
    float mx = camera.x + camera.gazeX * camera.distance;
    float my = camera.y + camera.gazeY * camera.distance;
    float mz = camera.z + camera.gazeZ * camera.distance;

    // Compute u = gaze x v
    float ux = camera.gazeY * camera.upZ - (camera.gazeZ * camera.upY);
    float uy = camera.gazeZ * camera.upX - (camera.gazeX * camera.upZ);
    float uz = camera.gazeX * camera.upY - (camera.gazeY * camera.upX);

    // Compute q = m + lu + tv
    float qx = mx + camera.left * ux + camera.top * camera.upX;
    float qy = my + camera.left * uy + camera.top * camera.upY;
    float qz = mz + camera.left * uz + camera.top * camera.upZ;

    // Compute su and sv
    float su = (camera.right - camera.left) * (i + 0.5f) / camera.width;
    float sv = (camera.top - camera.bottom) * (j + 0.5f) / camera.height;
    
    // Compute s = q + suu - svv
    *sx = qx + su * ux - sv * camera.upX;
    *sy = qy + su * uy - sv * camera.upY;
    *sz = qz + su * uz - sv * camera.upZ;
    
    float dx = *sx - camera.x;
    float dy = *sy - camera.y;
    float dz = *sz - camera.z;
    
    float length = sqrt(dx * dx + dy * dy + dz * dz);
        
    if (length != 0.0f) {
            dx /= length;
            dy /= length;
            dz /= length;
    }

    if (computeSphereIntersection(vertexList, spheres, materials, camera.x, camera.y, camera.z, dx, dy, dz, &t1, material1, center)){
        if (t1 < min_t) {
            min_t = t1;
            min_index = 0;
        }
    }

    if (computeTriangleIntersection(vertexList, triangles, materials, camera.x, camera.y, camera.z, dx, dy, dz, &t2, material2, v0, v1, v2)){
        if (t2 < min_t) {
            min_t = t2;
            min_index = 1;
        }
    }

    if (computeMeshIntersection(vertexList, meshes, materials, camera.x, camera.y, camera.z, dx, dy, dz, &t3, material3, v0, v1, v2)){
        if (t3 < min_t) {
            min_t = t3;
            min_index = 2;
        }
    }

    if (min_index == 0) {
        ray[0] = camera.x + (dx) * min_t;
        ray[1] = camera.y + (dy) * min_t;
        ray[2] = camera.z + (dz) * min_t;

        normal.x = (ray[0] - center.x);
        normal.y = (ray[1] - center.y);
        normal.z = (ray[2] - center.z);
        
        float length = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        
        if (length != 0.0f) {
            normal.x /= length;
            normal.y /= length;
            normal.z /= length;
        }
        
        material=material1;
    } 

    else if (min_index == 1) {
        ray[0] = camera.x + (dx) * min_t;
        ray[1] = camera.y + (dy) * min_t;
        ray[2] = camera.z + (dz) * min_t;

        Vertex e1 = {v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
        Vertex e2 = {v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};

        normal.x = e1.y * e2.z - e1.z * e2.y;
        normal.y = e1.z * e2.x - e1.x * e2.z;
        normal.z = e1.x * e2.y - e1.y * e2.x;

        float length = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        
        if (length != 0.0f) {
            normal.x /= length;
            normal.y /= length;
            normal.z /= length;
        }

        material=material2;
    } 

    else if (min_index == 2) {
        ray[0] = camera.x + (dx) * min_t;
        ray[1] = camera.y + (dy) * min_t;
        ray[2] = camera.z + (dz) * min_t;

        Vertex e1 = {v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
        Vertex e2 = {v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};
        
        float length = std::sqrt(e1.x * e1.x + e1.y * e1.y + e1.z * e1.z);
        e1.x /= length;
        e1.y /= length;
        e1.z /= length;

        length = std::sqrt(e2.x * e2.x + e2.y * e2.y + e2.z * e2.z);
        e2.x /= length;
        e2.y /= length;
        e2.z /= length;

        normal.x = e1.y * e2.z - e1.z * e2.y;
        normal.y = e1.z * e2.x - e1.x * e2.z;
        normal.z = e1.x * e2.y - e1.y * e2.x;

        length = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        
        if (length != 0.0f) {
            normal.x /= length;
            normal.y /= length;
            normal.z /= length;
        }

        material=material3;
    } 

    else {
        ray[0] = backgroundColor.r;
        ray[1] = backgroundColor.g;
        ray[2] = backgroundColor.b;
    }
    
    return std::make_tuple(ray, material);
}

bool computeShadow(std::vector<Vertex> vertexList,std::vector<Material> materials,std::vector<std::tuple<int, int, int, float>> spheres,std::vector<std::tuple<int, int, int, int, int>> triangles,std::vector<std::tuple<int, int, std::vector<int>>> meshes,std::vector<PointLight> pointLights, float shadowRayEpsilon, std::vector<float> ray){

    Vertex center, v0, v1, v2;
    Material material;
    float t= std::numeric_limits<float>::infinity();

    bool shadow = false;

    for (const auto& light : pointLights) {

        float shadowRayX = light.x - ray[0];
        float shadowRayY = light.y - ray[1];
        float shadowRayZ = light.z - ray[2];
        float shadowRayDistance = std::sqrt(shadowRayX * shadowRayX + shadowRayY * shadowRayY + shadowRayZ * shadowRayZ);
        
        shadowRayX /= shadowRayDistance;
        shadowRayY /= shadowRayDistance;
        shadowRayZ /= shadowRayDistance;

        //(x+wi)*epislon
        float shadowRayOriginX = ray[0] + shadowRayX * shadowRayEpsilon; 
        float shadowRayOriginY = ray[1] + shadowRayY * shadowRayEpsilon; 
        float shadowRayOriginZ = ray[2] + shadowRayZ * shadowRayEpsilon; 

        if (computeSphereIntersection (vertexList, spheres, materials, shadowRayOriginX, shadowRayOriginY, shadowRayOriginZ, shadowRayX, shadowRayY, shadowRayZ, &t, material, center) 
        || (computeTriangleIntersection (vertexList, triangles, materials, shadowRayOriginX, shadowRayOriginY, shadowRayOriginZ, shadowRayX, shadowRayY, shadowRayZ, &t, material, v0, v1, v2)) 
        || (computeMeshIntersection (vertexList, meshes, materials, shadowRayOriginX, shadowRayOriginY, shadowRayOriginZ, shadowRayX, shadowRayY, shadowRayZ, &t, material, v0, v1, v2))) {
            shadow = true;
        }

    }
    return shadow;
}

Color computeLighting(std::vector<Vertex> vertexList,std::vector<Material> materials,std::vector<std::tuple<int, int, int, float>> spheres,std::vector<std::tuple<int, int, int, int, int>> triangles,std::vector<std::tuple<int, int, std::vector<int>>> meshes,std::vector<PointLight> pointLights, float shadowRayEpsilon,Camera camera,Color ambientLight, Material& material, std::vector<float> ray, Vertex& normal) {

    Vertex viewDirection = {camera.x - ray[0] , camera.y - ray[1], camera.z - ray[2]};
    Color lighting={0.0f,0.0f,0.0f};
    
    float lightIntensity_R, lightIntensity_G, lightIntensity_B;

    float ambientFunc_r = (material.ambient.r) * (ambientLight.r / 255.0f);
    float ambientFunc_g = (material.ambient.g) * (ambientLight.g / 255.0f);
    float ambientFunc_b = (material.ambient.b) * (ambientLight.b / 255.0f);

    for (const auto& light : pointLights) {
        
        if (computeShadow(vertexList,materials,spheres,triangles,meshes,pointLights,shadowRayEpsilon, ray)) {
            lighting.r = ambientFunc_r;
            lighting.g = ambientFunc_g;
            lighting.b = ambientFunc_b;
            return lighting;
        }

        float lightDirX = (light.x - ray[0]);
        float lightDirY = (light.y - ray[1]);
        float lightDirZ = (light.z - ray[2]);

        float lightDistance = std::sqrt(lightDirX * lightDirX + lightDirY * lightDirY + lightDirZ * lightDirZ);
        
        if(lightDistance != 0){
            lightDirX /= lightDistance;
            lightDirY /= lightDistance;
            lightDirZ /= lightDistance;
        }

        lightIntensity_R = light.intensity.r / 255.0f;
        lightIntensity_G = light.intensity.g / 255.0f;
        lightIntensity_B = light.intensity.b / 255.0f;

        float diffuseDot = std::max(0.0f, normal.x * lightDirX + normal.y * lightDirY + normal.z * lightDirZ);
       
        float diffuseFunc_r = material.diffuse.r * (lightIntensity_R / (lightDistance * lightDistance)) * diffuseDot;
        float diffuseFunc_g = material.diffuse.g * (lightIntensity_G / (lightDistance * lightDistance)) * diffuseDot;
        float diffuseFunc_b = material.diffuse.b * (lightIntensity_B / (lightDistance * lightDistance)) * diffuseDot;
        float viewDirectionDistance = std::sqrt(viewDirection.x * viewDirection.x + viewDirection.y * viewDirection.y + viewDirection.z * viewDirection.z);
        
        viewDirection.x /= viewDirectionDistance;
        viewDirection.y /= viewDirectionDistance;
        viewDirection.z /= viewDirectionDistance;
        
        Vertex halfVec = {(lightDirX + viewDirection.x), (lightDirY + viewDirection.y), (lightDirZ + viewDirection.z)};
        float halfVecLength = std::sqrt(halfVec.x * halfVec.x + halfVec.y * halfVec.y + halfVec.z * halfVec.z);
        halfVec.x /= halfVecLength;
        halfVec.y /= halfVecLength;
        halfVec.z /= halfVecLength;

        // // Calculate specular dot product
        float specularDot = std::max(0.0f, normal.x * halfVec.x + normal.y * halfVec.y + normal.z * halfVec.z);

        specularDot = std::pow(specularDot, material.specularExponent);
        float specularFunc_r = material.specular.r * specularDot * (lightIntensity_R / (lightDistance * lightDistance));
        float specularFunc_g = material.specular.g * specularDot * (lightIntensity_G / (lightDistance * lightDistance));
        float specularFunc_b = material.specular.b * specularDot * (lightIntensity_B / (lightDistance * lightDistance));

        lighting.r += diffuseFunc_r + specularFunc_r;
        lighting.g += diffuseFunc_g + specularFunc_g;
        lighting.b += diffuseFunc_b + specularFunc_b;
    }

    lighting.r += ambientFunc_r;
    lighting.g += ambientFunc_g;
    lighting.b += ambientFunc_b; 

    lighting.r = std::min(1.0f, std::max(0.0f, lighting.r));
    lighting.g = std::min(1.0f, std::max(0.0f, lighting.g));
    lighting.b = std::min(1.0f, std::max(0.0f, lighting.b));

    return lighting;
}

int main(int argc, char* argv[]) {
    std::string inputFilename = argv[1];
    std::ofstream outputFile("output.ppm", std::ios::binary);

    Camera camera = parseCamera(inputFilename);
    Color backgroundColor = parseBackgroundColor(inputFilename);
    std::vector<Vertex> vertexList = parseVertexList(inputFilename);
    std::vector<std::tuple<int, int, int, float>> spheres = parseSpheres(inputFilename);
    std::vector<Material> materials = parseMaterials(inputFilename);
    std::vector<std::tuple<int, int, int, int, int>> triangles = parseTriangles(inputFilename);
    std::vector<PointLight> pointLights = parsePointLights(inputFilename);
    Color ambientLight = parseAmbientLight(inputFilename);
    float shadowRayEpsilon = parseShadowRayEpsilon(inputFilename);
    std::vector<std::tuple<int, int, std::vector<int>>> meshes = parseMeshes(inputFilename);

    float sx, sy, sz;
    Vertex normal = {0, 0, 0};
    if (outputFile.is_open()) {
        outputFile << "P6\n" << camera.width << " " << camera.height << "\n255\n";

        for (int j = 0; j < camera.height; j++) {
            for (int i = 0; i < camera.width; i++) {
                std::tuple<std::vector<float>, Material> rayAndMaterial = computeCameraRay(camera, backgroundColor, vertexList, spheres, triangles, meshes, materials, i, j, &sx, &sy, &sz, normal);
                std::vector<float> ray = std::get<0>(rayAndMaterial);
                Material material = std::get<1>(rayAndMaterial);
                Color pixelColor;

                if (ray[0] == backgroundColor.r && ray[1] == backgroundColor.g && ray[2] == backgroundColor.b) {
                    pixelColor = backgroundColor;
                } else {
                    Color lighting = computeLighting(vertexList, materials, spheres, triangles, meshes, pointLights, shadowRayEpsilon, camera, ambientLight, material, ray, normal);
                    pixelColor.r = lighting.r;
                    pixelColor.g = lighting.g;
                    pixelColor.b = lighting.b;
                }
                unsigned char r = (unsigned char)(pixelColor.r * 255);
                unsigned char g = (unsigned char)(pixelColor.g * 255);
                unsigned char b = (unsigned char)(pixelColor.b * 255);

                outputFile << r << g << b;
            }
        }

        outputFile.close();
        std::cout << "Output file 'output.ppm' generated successfully." << std::endl;
    } else {
        std::cerr << "Error: Unable to open output file." << std::endl;
    }
    return 0;
}
