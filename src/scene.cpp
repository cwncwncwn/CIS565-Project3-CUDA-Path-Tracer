#define TINYOBJLOADER_IMPLEMENTATION
#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
#include "tiny_obj_loader.h"
#include "ImGui/imgui.h"

using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.0;
        }
        else if (p["TYPE"] == "Transmissive") {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0;
            newMaterial.indexOfRefraction = 1.55;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        newGeom.custom_mesh_idx = glm::vec2(-1.0);
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else {
            newGeom.type = CUSTOM;
            tinyobj::ObjReaderConfig reader_config;
            tinyobj::ObjReader reader;

            // Parse to get object name and read object
            std::size_t pos = jsonName.rfind("/");  // Find the last '/'
            std::string objName = jsonName.substr(0, pos + 1) + "Mesh/" + p["PATH"].get<std::string>() + ".obj";
            if (!reader.ParseFromFile(objName)) {
                if (!reader.Error().empty()) {
                    std::cerr << "Error: " << reader.Error() << std::endl;
                }
                exit(1);
            }

            // Get object data
            auto& attrib = reader.GetAttrib();
            auto& shapes = reader.GetShapes();
            auto& uv = reader.GetMaterials();

            // for each shape:
            //  record index of the mesh in geoms
            //  for each index:
            //      push back each vertex based on its index (record pos, nor, uv)
            newGeom.custom_mesh_idx.x = this->custom_meshes.size();
            for (int shapeIdx = 0; shapeIdx < shapes.size(); shapeIdx++) {
                // record mesh index in geoms                
                // record vertices
                Custom_Mesh custom_mesh;
                auto& currMesh = shapes[shapeIdx].mesh;
                custom_mesh.vertex_indices.x = this->vertices.size();

                for (size_t idx = 0; idx < currMesh.indices.size(); idx++) {
                    Vertex v;
                    for (int i = 0; i < 3; i++) {
                        v.pos[i] = attrib.vertices[currMesh.indices[idx].vertex_index * 3 + i];
                        v.normal[i] = attrib.normals[currMesh.indices[idx].normal_index * 3 + i];
                        if (i < 2) {
                            v.uv[i] = attrib.texcoords[currMesh.indices[idx].texcoord_index * 2 + i];
                        }
                    }
                    this->vertices.push_back(v);
                }
                custom_mesh.vertex_indices.y = this->vertices.size() - 1;
                custom_mesh.mesh_indices.x = this->geoms.size();
                custom_mesh.mesh_indices.y = shapeIdx;
                this->custom_meshes.push_back(custom_mesh);
            }
            newGeom.custom_mesh_idx.y = this->custom_meshes.size() - 1;
        }

        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
