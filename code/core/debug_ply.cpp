#include "debug_ply.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace SilkRAD::Core::Debug {
    namespace {
        struct Color24 {
            uint8_t r = 0;
            uint8_t g = 0;
            uint8_t b = 0;
        };

        struct PlyVertex {
            Common::Vec3f pos;
            Color24 color;
        };

        struct PlyFace {
            size_t i0 = 0;
            size_t i1 = 0;
            size_t i2 = 0;
        };

        float channel_from_rgbexp(uint8_t channel, int8_t exponent)
        {
            return static_cast<float>(channel) * std::ldexp(1.0f, exponent);
        }

        Color24 color_from_rgbexp32(const ::BSP::RGBExp32& sample)
        {
            const float r = std::max(0.0f, channel_from_rgbexp(sample.r, sample.exp));
            const float g = std::max(0.0f, channel_from_rgbexp(sample.g, sample.exp));
            const float b = std::max(0.0f, channel_from_rgbexp(sample.b, sample.exp));
            const float maxChannel = std::max(r, std::max(g, b));
            const float scale = maxChannel > 255.0f ? (255.0f / maxChannel) : 1.0f;

            Color24 color;
            color.r = static_cast<uint8_t>(std::clamp(r * scale, 0.0f, 255.0f));
            color.g = static_cast<uint8_t>(std::clamp(g * scale, 0.0f, 255.0f));
            color.b = static_cast<uint8_t>(std::clamp(b * scale, 0.0f, 255.0f));
            return color;
        }

        bool lightmap_base_index(
            const ::BSP::DFace& dface,
            size_t& outBaseIndex
        )
        {
            if (dface.lightOffset < 0) {
                return false;
            }

            outBaseIndex = static_cast<size_t>(dface.lightOffset / sizeof(::BSP::RGBExp32));
            return true;
        }

        template<typename LuxelContainer>
        void append_luxel_grid(
            std::vector<PlyVertex>& outVertices,
            std::vector<PlyFace>& outFaces,
            const LuxelContainer& luxels,
            size_t width,
            size_t height,
            const std::vector<::BSP::RGBExp32>& lightSamples,
            size_t lightmapBaseIndex
        )
        {
            if (width < 2 || height < 2) {
                return;
            }

            if (luxels.size() < width * height) {
                return;
            }

            if (lightmapBaseIndex + (width * height) > lightSamples.size()) {
                return;
            }

            const size_t vertexBase = outVertices.size();
            outVertices.reserve(outVertices.size() + width * height);
            outFaces.reserve(outFaces.size() + (width - 1) * (height - 1) * 2);

            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    const size_t index = y * width + x;
                    outVertices.push_back(PlyVertex{
                        luxels[index].pos,
                        color_from_rgbexp32(lightSamples[lightmapBaseIndex + index])
                    });
                }
            }

            for (size_t y = 0; y + 1 < height; ++y) {
                for (size_t x = 0; x + 1 < width; ++x) {
                    const size_t i0 = vertexBase + y * width + x;
                    const size_t i1 = vertexBase + (y + 1) * width + x;
                    const size_t i2 = vertexBase + (y + 1) * width + (x + 1);
                    const size_t i3 = vertexBase + y * width + (x + 1);
                    outFaces.push_back(PlyFace{ i0, i1, i2 });
                    outFaces.push_back(PlyFace{ i0, i2, i3 });
                }
            }
        }

        void append_face_lightmap_mesh(
            std::vector<PlyVertex>& outVertices,
            std::vector<PlyFace>& outFaces,
            const ::BSP::BSP& bsp,
            const RuntimeState& state
        )
        {
            const std::vector<::BSP::RGBExp32>& lightSamples = bsp.get_lightsamples();
            const size_t faceCount = bsp.get_dfaces().size();
            for (size_t faceIndex = 0; faceIndex < faceCount; ++faceIndex) {
                const ::BSP::DFace& dface = bsp.get_dfaces()[faceIndex];
                size_t lightmapBaseIndex = 0;
                if (!lightmap_base_index(dface, lightmapBaseIndex)) {
                    continue;
                }

                const size_t width = static_cast<size_t>(dface.lightmapTextureSizeInLuxels[0] + 1);
                const size_t height = static_cast<size_t>(dface.lightmapTextureSizeInLuxels[1] + 1);
                if (width == 0 || height == 0) {
                    continue;
                }

                const Geometry::DispGeometry& dispGeometry = state.dispGeometry[faceIndex];
                if (dispGeometry.valid) {
                    append_luxel_grid(
                        outVertices,
                        outFaces,
                        dispGeometry.luxels,
                        width,
                        height,
                        lightSamples,
                        lightmapBaseIndex
                    );
                    continue;
                }

                const Geometry::FaceGeometry& faceGeometry = state.faceGeometry[faceIndex];
                if (!faceGeometry.valid || faceGeometry.isDisplacement) {
                    continue;
                }

                append_luxel_grid(
                    outVertices,
                    outFaces,
                    faceGeometry.luxels.values,
                    width,
                    height,
                    lightSamples,
                    lightmapBaseIndex
                );
            }
        }

        std::string blender_script_filename(const std::string& plyFilename)
        {
            const size_t dot = plyFilename.find_last_of('.');
            if (dot == std::string::npos) {
                return plyFilename + ".blender.py";
            }
            return plyFilename.substr(0, dot) + ".blender.py";
        }

        void write_blender_import_script(
            const ::BSP::BSP& bsp,
            const std::string& plyFilename
        )
        {
            const std::string scriptFilename = blender_script_filename(plyFilename);
            std::ofstream out(scriptFilename, std::ios::binary);
            if (!out) {
                return;
            }

            bool haveSun = false;
            float sunDirX = 0.0f;
            float sunDirY = 0.0f;
            float sunDirZ = -1.0f;
            float sunEnergy = 3.0f;
            for (const ::BSP::DWorldLight& light : bsp.get_worldlights()) {
                if (light.type != ::BSP::EMIT_SKYLIGHT || light.style != 0) {
                    continue;
                }

                const float len = std::sqrt(
                    light.normal.x * light.normal.x
                    + light.normal.y * light.normal.y
                    + light.normal.z * light.normal.z
                );
                if (len <= 1e-6f) {
                    continue;
                }

                haveSun = true;
                sunDirX = -light.normal.x / len;
                sunDirY = -light.normal.y / len;
                sunDirZ = -light.normal.z / len;
                sunEnergy = std::max(0.5f, std::max(light.intensity.x, std::max(light.intensity.y, light.intensity.z)) * 0.05f);
                break;
            }

            out << "import bpy\n";
            out << "import mathutils\n\n";
            out << "PLY_PATH = r'''" << plyFilename << "'''\n";
            out << "MATERIAL_NAME = 'SilkRADLightingDebug'\n";
            out << "SUN_NAME = 'SilkRAD_Sun'\n\n";
            out << "before = set(bpy.data.objects)\n";
            out << "if hasattr(bpy.ops.wm, 'ply_import'):\n";
            out << "    bpy.ops.wm.ply_import(filepath=PLY_PATH)\n";
            out << "else:\n";
            out << "    bpy.ops.import_mesh.ply(filepath=PLY_PATH)\n";
            out << "imported = [obj for obj in bpy.data.objects if obj not in before]\n\n";
            out << "mat = bpy.data.materials.get(MATERIAL_NAME)\n";
            out << "if mat is None:\n";
            out << "    mat = bpy.data.materials.new(MATERIAL_NAME)\n";
            out << "mat.use_nodes = True\n";
            out << "nodes = mat.node_tree.nodes\n";
            out << "links = mat.node_tree.links\n";
            out << "nodes.clear()\n";
            out << "out_node = nodes.new('ShaderNodeOutputMaterial')\n";
            out << "bsdf = nodes.new('ShaderNodeBsdfPrincipled')\n";
            out << "attr = nodes.new('ShaderNodeAttribute')\n";
            out << "attr.location = (-300, 0)\n";
            out << "bsdf.location = (0, 0)\n";
            out << "out_node.location = (250, 0)\n";
            out << "links.new(attr.outputs['Color'], bsdf.inputs['Base Color'])\n";
            out << "links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])\n\n";
            out << "for obj in imported:\n";
            out << "    if obj.type != 'MESH':\n";
            out << "        continue\n";
            out << "    color_attrs = getattr(obj.data, 'color_attributes', None)\n";
            out << "    if color_attrs and len(color_attrs) > 0:\n";
            out << "        attr.attribute_name = color_attrs[0].name\n";
            out << "    elif hasattr(obj.data, 'vertex_colors') and len(obj.data.vertex_colors) > 0:\n";
            out << "        attr.attribute_name = obj.data.vertex_colors[0].name\n";
            out << "    if obj.data.materials:\n";
            out << "        obj.data.materials[0] = mat\n";
            out << "    else:\n";
            out << "        obj.data.materials.append(mat)\n\n";

            if (haveSun) {
                out << "sun = bpy.data.objects.get(SUN_NAME)\n";
                out << "if sun is None:\n";
                out << "    light_data = bpy.data.lights.new(SUN_NAME, type='SUN')\n";
                out << "    sun = bpy.data.objects.new(SUN_NAME, light_data)\n";
                out << "    bpy.context.scene.collection.objects.link(sun)\n";
                out << "direction = mathutils.Vector((" << sunDirX << ", " << sunDirY << ", " << sunDirZ << "))\n";
                out << "sun.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()\n";
                out << "sun.data.energy = " << sunEnergy << "\n";
            }
        }
    }

    void export_lighting_receivers_ply(
        const ::BSP::BSP& bsp,
        const RuntimeState& state,
        const std::string& filename
    )
    {
        std::vector<PlyVertex> vertices;
        std::vector<PlyFace> faces;
        append_face_lightmap_mesh(vertices, faces, bsp, state);

        std::ofstream out(filename, std::ios::binary);
        if (!out) {
            return;
        }

        out << "ply\n";
        out << "format ascii 1.0\n";
        out << "comment SilkRAD core lightmap debug\n";
        out << "element vertex " << vertices.size() << "\n";
        out << "property float x\n";
        out << "property float y\n";
        out << "property float z\n";
        out << "property uchar red\n";
        out << "property uchar green\n";
        out << "property uchar blue\n";
        out << "element face " << faces.size() << "\n";
        out << "property list uchar int vertex_indices\n";
        out << "end_header\n";

        for (const PlyVertex& v : vertices) {
            out << v.pos.x << ' ' << v.pos.y << ' ' << v.pos.z << ' '
                << static_cast<int>(v.color.r) << ' '
                << static_cast<int>(v.color.g) << ' '
                << static_cast<int>(v.color.b) << '\n';
        }

        for (const PlyFace& f : faces) {
            out << "3 " << f.i0 << ' ' << f.i1 << ' ' << f.i2 << '\n';
        }

        write_blender_import_script(bsp, filename);
    }
}
