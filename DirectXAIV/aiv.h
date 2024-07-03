#pragma once
#include <dxgi1_4.h>
#include <d3d12.h>

#include <iostream>
#include <vector>
#include <string>

#include <comdef.h>

#include <fstream>

#define SDL_MAIN_HANDLED 1
#include <SDL.h>
#include <SDL_syswm.h>

#define COMPTR(x) __uuidof(decltype(x)), reinterpret_cast<void**>(&x)

#define COMCHECK(x) {\
	HRESULT result = x;\
	if (result != S_OK)\
	{\
		_com_error err(result);\
		OutputDebugStringW((L"Error: " + std::wstring(err.ErrorMessage()) + L"\n").c_str());\
		throw std::exception();\
	}\
}

#include <DirectXMath.h>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#include "tiny_gltf.h"

namespace aiv
{
	struct Context
	{
		ID3D12Device* device;
		ID3D12CommandQueue* queue;
		ID3D12CommandAllocator* command_allocator;
		ID3D12GraphicsCommandList* command_list;
		ID3D12Fence* fence;
		UINT64 fence_value;
		HANDLE fence_event;

		~Context()
		{
			if (fence_event)
			{
				CloseHandle(fence_event);
			}

			if (fence)
			{
				fence->Release();
			}

			if (command_list)
			{
				command_list->Release();
			}

			if (command_allocator)
			{
				command_allocator->Release();
			}

			if (queue)
			{
				queue->Release();
			}

			if (device)
			{
				device->Release();
			}
		}
	};

	class DescriptorHeap
	{
	public:
		DescriptorHeap() = delete;
		DescriptorHeap(ID3D12Device* device, const D3D12_DESCRIPTOR_HEAP_TYPE descriptor_heap_type, const bool shader_visible)
		{
			descriptor_heap = nullptr;

			D3D12_DESCRIPTOR_HEAP_DESC descriptor_heap_desc = {};
			descriptor_heap_desc.Type = descriptor_heap_type;
			descriptor_heap_desc.NumDescriptors = 64;
			descriptor_heap_desc.Flags = shader_visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

			COMCHECK(device->CreateDescriptorHeap(&descriptor_heap_desc, COMPTR(descriptor_heap)));

			descriptor_heap_increment = device->GetDescriptorHandleIncrementSize(descriptor_heap_type);

			cpu_start_address = descriptor_heap->GetCPUDescriptorHandleForHeapStart();
			gpu_start_address = descriptor_heap->GetGPUDescriptorHandleForHeapStart();
		}

		DescriptorHeap(const DescriptorHeap& other) = delete;
		DescriptorHeap& operator=(const DescriptorHeap& other) = delete;
		DescriptorHeap(DescriptorHeap&& other) = delete;
		DescriptorHeap& operator=(DescriptorHeap&& other) = delete;

		~DescriptorHeap()
		{
			if (descriptor_heap)
			{
				descriptor_heap->Release();
			}
		}

		D3D12_CPU_DESCRIPTOR_HANDLE GetCPUHandle(const size_t index) const
		{
			D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle;
			cpu_handle.ptr = cpu_start_address.ptr + (index * descriptor_heap_increment);
			return cpu_handle;
		}

		D3D12_GPU_DESCRIPTOR_HANDLE GetGPUHandle(const size_t index) const
		{
			D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle;
			gpu_handle.ptr = gpu_start_address.ptr + (index * descriptor_heap_increment);
			return gpu_handle;
		}

		ID3D12DescriptorHeap* GetComPtr() const
		{
			return descriptor_heap;
		}

	protected:
		ID3D12DescriptorHeap* descriptor_heap;
		UINT descriptor_heap_increment;
		D3D12_CPU_DESCRIPTOR_HANDLE cpu_start_address;
		D3D12_GPU_DESCRIPTOR_HANDLE gpu_start_address;
	};

	ID3D12Resource* create_buffer(ID3D12Device* device, const UINT64 size, const D3D12_HEAP_TYPE heap_type)
	{
		ID3D12Resource* resource = nullptr;
		D3D12_HEAP_PROPERTIES heap_properties = {};
		heap_properties.Type = heap_type;

		D3D12_RESOURCE_DESC resource_desc = {};
		resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		resource_desc.Width = size;
		resource_desc.Height = 1;
		resource_desc.DepthOrArraySize = 1;
		resource_desc.MipLevels = 1;
		resource_desc.SampleDesc.Count = 1;
		resource_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		if (heap_type == D3D12_HEAP_TYPE_DEFAULT)
		{
			resource_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
		}

		D3D12_RESOURCE_ALLOCATION_INFO allocation_info = device->GetResourceAllocationInfo(0, 1, &resource_desc);

		COMCHECK(device->CreateCommittedResource(&heap_properties,
			D3D12_HEAP_FLAG_NONE,
			&resource_desc,
			heap_type == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			COMPTR(resource)));

		return resource;
	}

	ID3D12Resource* create_texture2d(ID3D12Device* device, const UINT width, const UINT height, const DXGI_FORMAT pixel_format, const wchar_t* name)
	{
		ID3D12Resource* resource = nullptr;
		D3D12_HEAP_PROPERTIES heap_properties = {};
		heap_properties.Type = D3D12_HEAP_TYPE_DEFAULT;

		D3D12_RESOURCE_DESC resource_desc = {};
		resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		resource_desc.Format = pixel_format;
		resource_desc.Width = width;
		resource_desc.Height = height;
		resource_desc.DepthOrArraySize = 1;
		resource_desc.MipLevels = 1;
		resource_desc.SampleDesc.Count = 1;
		resource_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
		resource_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

		D3D12_RESOURCE_ALLOCATION_INFO allocation_info = device->GetResourceAllocationInfo(0, 1, &resource_desc);

		D3D12_CLEAR_VALUE clear_value = {};
		clear_value.Format = pixel_format;

		COMCHECK(device->CreateCommittedResource(&heap_properties,
			D3D12_HEAP_FLAG_NONE,
			&resource_desc,
			D3D12_RESOURCE_STATE_COPY_SOURCE,
			&clear_value,
			COMPTR(resource)));

		resource->SetName(name);

		return resource;
	}

	ID3D12Resource* create_depth(ID3D12Device* device, const UINT width, const UINT height, const wchar_t* name)
	{
		ID3D12Resource* resource = nullptr;
		D3D12_HEAP_PROPERTIES heap_properties = {};
		heap_properties.Type = D3D12_HEAP_TYPE_DEFAULT;

		D3D12_RESOURCE_DESC resource_desc = {};
		resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		resource_desc.Format = DXGI_FORMAT_D32_FLOAT;
		resource_desc.Width = width;
		resource_desc.Height = height;
		resource_desc.DepthOrArraySize = 1;
		resource_desc.MipLevels = 1;
		resource_desc.SampleDesc.Count = 1;
		resource_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
		resource_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

		D3D12_RESOURCE_ALLOCATION_INFO allocation_info = device->GetResourceAllocationInfo(0, 1, &resource_desc);

		D3D12_CLEAR_VALUE clear_value = {};
		clear_value.Format = DXGI_FORMAT_D32_FLOAT;
		clear_value.DepthStencil.Depth = 1;

		COMCHECK(device->CreateCommittedResource(&heap_properties,
			D3D12_HEAP_FLAG_NONE,
			&resource_desc,
			D3D12_RESOURCE_STATE_COPY_SOURCE,
			&clear_value,
			COMPTR(resource)));

		resource->SetName(name);

		return resource;
	}

	void copy_resource(Context& ctx, ID3D12Resource* src, ID3D12Resource* dst,
		const D3D12_RESOURCE_STATES src_state_before, const D3D12_RESOURCE_STATES src_state_after, const D3D12_RESOURCE_STATES dst_state_before, const D3D12_RESOURCE_STATES dst_state_after)
	{
		ctx.command_allocator->Reset();
		ctx.command_list->Reset(ctx.command_allocator, nullptr);

		D3D12_RESOURCE_BARRIER barriers[2] = {};
		barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barriers[0].Transition.pResource = src;
		barriers[0].Transition.StateBefore = src_state_before;
		barriers[0].Transition.StateAfter = src_state_after;
		barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barriers[1].Transition.pResource = dst;
		barriers[1].Transition.StateBefore = dst_state_before;
		barriers[1].Transition.StateAfter = dst_state_after;

		// barrier before
		ctx.command_list->ResourceBarrier(2, barriers);

		ctx.command_list->CopyResource(dst, src);

		barriers[0].Transition.StateBefore = src_state_after;
		barriers[0].Transition.StateAfter = src_state_before;
		barriers[1].Transition.StateBefore = dst_state_after;
		barriers[1].Transition.StateAfter = dst_state_before;
		// barrier after
		ctx.command_list->ResourceBarrier(2, barriers);

		ctx.command_list->Close();

		ctx.queue->ExecuteCommandLists(1, reinterpret_cast<ID3D12CommandList**>(&ctx.command_list));

		ctx.queue->Signal(ctx.fence, ++ctx.fence_value);
		ctx.fence->SetEventOnCompletion(ctx.fence_value, ctx.fence_event);

		WaitForSingleObject(ctx.fence_event, INFINITE);
	}

	std::vector<char> load_file(const char* filename)
	{
		std::ifstream file;
		file.open(filename, std::ios::binary);
		file.seekg(0, std::ios::end);
		std::vector<char>data(file.tellg());
		file.seekg(0, std::ios::beg);

		file.read(data.data(), data.size());
		file.close();

		return data;
	}

	template<typename T>
	void append_subobject_to_stream(std::vector<char>& stream, const D3D12_PIPELINE_STATE_SUBOBJECT_TYPE subobject_type, const T& subobject)
	{
		const size_t ptr_size = sizeof(void*);

		if (stream.size() % ptr_size != 0)
		{
			stream.resize(stream.size() + (ptr_size - (stream.size() % ptr_size)));
		}

		std::vector<char> subobject_type_data(reinterpret_cast<const char*>(&subobject_type), reinterpret_cast<const char*>(&subobject_type) + sizeof(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE));
		stream.insert(stream.end(), subobject_type_data.begin(), subobject_type_data.end());

		if (stream.size() % alignof(T) != 0)
		{
			stream.resize(stream.size() + (alignof(T) - (stream.size() % alignof(T))));
		}

		std::vector<char> subobject_data(reinterpret_cast<const char*>(&subobject), reinterpret_cast<const char*>(&subobject) + sizeof(T));
		stream.insert(stream.end(), subobject_data.begin(), subobject_data.end());
	}

	std::vector<float> load_stl(const char* filename, uint32_t& number_of_triangles)
	{
		std::vector<char> data = load_file(filename);

		size_t offset = 80;
		const uint32_t* number_of_triangles_ptr = reinterpret_cast<const uint32_t*>(data.data() + offset);

		number_of_triangles = *number_of_triangles_ptr;

		offset += 4;

		std::vector<float> vertices;

		for (size_t i = 0; i < number_of_triangles; i++)
		{
			const float* n0 = reinterpret_cast<const float*>(data.data() + offset);
			const float* v0 = reinterpret_cast<const float*>(data.data() + offset + 12);
			const float* v1 = reinterpret_cast<const float*>(data.data() + offset + 12 + 12 + 12);
			const float* v2 = reinterpret_cast<const float*>(data.data() + offset + 12 + 12);

			vertices.push_back(v0[0]); vertices.push_back(v0[1]);  vertices.push_back(v0[2]);
			vertices.push_back(n0[0]); vertices.push_back(n0[1]);  vertices.push_back(n0[2]);
			vertices.push_back(v1[0]); vertices.push_back(v1[1]);  vertices.push_back(v1[2]);
			vertices.push_back(n0[0]); vertices.push_back(n0[1]);  vertices.push_back(n0[2]);
			vertices.push_back(v2[0]); vertices.push_back(v2[1]);  vertices.push_back(v2[2]);
			vertices.push_back(n0[0]); vertices.push_back(n0[1]);  vertices.push_back(n0[2]);

			offset += 50;
		}

		return vertices;
	}

	ID3D12Resource* create_buffer_from_data(Context& ctx, const void* ptr, const size_t size)
	{
		ID3D12Resource* upload_buffer = create_buffer(ctx.device, size, D3D12_HEAP_TYPE_UPLOAD);

		void* upload_data;
		upload_buffer->Map(0, nullptr, &upload_data);

		::memcpy(upload_data, ptr, size);

		upload_buffer->Unmap(0, nullptr);

		ID3D12Resource* default_buffer = create_buffer(ctx.device, size, D3D12_HEAP_TYPE_DEFAULT);

		copy_resource(ctx, upload_buffer, default_buffer, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);

		upload_buffer->Release();

		return default_buffer;
	}

	ID3D12Resource* create_texture2d_from_data(Context& ctx, const uint32_t width, const uint32_t height, const std::vector<unsigned char>& data)
	{
		ID3D12Resource* texture = aiv::create_texture2d(ctx.device, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, L"Texture");
		D3D12_RESOURCE_DESC texture_desc = texture->GetDesc();
		ID3D12Resource* src_buffer = aiv::create_buffer_from_data(ctx, data.data(), data.size());

		ctx.command_allocator->Reset();
		ctx.command_list->Reset(ctx.command_allocator, nullptr);

		D3D12_TEXTURE_COPY_LOCATION src_location = {};
		src_location.pResource = src_buffer;
		src_location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
		ctx.device->GetCopyableFootprints(&texture_desc, 0, 1, 0, &src_location.PlacedFootprint, nullptr, nullptr, nullptr);

		D3D12_TEXTURE_COPY_LOCATION dst_location = {};
		dst_location.pResource = texture;
		dst_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;

		ctx.command_list->CopyTextureRegion(&dst_location, 0, 0, 0, &src_location, nullptr);

		ctx.command_list->Close();

		ctx.queue->ExecuteCommandLists(1, reinterpret_cast<ID3D12CommandList**>(&ctx.command_list));

		ctx.queue->Signal(ctx.fence, ++ctx.fence_value);
		ctx.fence->SetEventOnCompletion(ctx.fence_value, ctx.fence_event);

		WaitForSingleObject(ctx.fence_event, INFINITE);

		src_buffer->Release();

		return texture;
	}

	class Actor
	{
	public:
		Actor()
		{
			static size_t counter = 0;

			descriptor_heap_base = counter;

			counter += 8;

			scaling = DirectX::XMMatrixIdentity();
			rotation = DirectX::XMMatrixIdentity();
			location = DirectX::XMMatrixIdentity();
		}

		template<typename T>
		std::vector<T> LoadDataFromGLTFAccessor(tinygltf::Model& model, const int accessor_id)
		{
			auto accessor = model.accessors[accessor_id];

			auto buffer_view_id = accessor.bufferView;

			const uint32_t accessor_byte_offset = accessor.byteOffset;
			const uint32_t accessor_count = accessor.count;

			const uint32_t buffer_view_offset = model.bufferViews[buffer_view_id].byteOffset;
			const uint32_t buffer_view_size = model.bufferViews[buffer_view_id].byteLength;

			auto buffer_id = model.bufferViews[buffer_view_id].buffer;

			auto buffer = model.buffers[buffer_id].data;

			const T* indices_data_start = reinterpret_cast<const T*>(buffer.data() + buffer_view_offset);
			const T* indices_data_end = reinterpret_cast<const T*>(buffer.data() + buffer_view_offset + buffer_view_size);

			return std::vector<T>(indices_data_start, indices_data_end);
		}

		ID3D12Resource* create_blas(Context& ctx, ID3D12Resource* index_buffer, const uint32_t num_indices, ID3D12Resource* vertex_buffer, const uint32_t num_vertices)
		{
			D3D12_RAYTRACING_GEOMETRY_DESC geometry_desc = {};
			geometry_desc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;

			geometry_desc.Triangles.IndexBuffer = index_buffer->GetGPUVirtualAddress();
			geometry_desc.Triangles.IndexFormat = DXGI_FORMAT_R16_UINT;
			geometry_desc.Triangles.IndexCount = num_indices;
			geometry_desc.Triangles.VertexBuffer.StartAddress = vertex_buffer->GetGPUVirtualAddress();
			geometry_desc.Triangles.VertexBuffer.StrideInBytes = 12;
			geometry_desc.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
			geometry_desc.Triangles.VertexCount = num_vertices;
			geometry_desc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

			D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC blas_desc = {};
			blas_desc.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
			blas_desc.Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
			blas_desc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
			blas_desc.Inputs.NumDescs = 1;
			blas_desc.Inputs.pGeometryDescs = &geometry_desc;

			D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO as_brebuild_info = {};

			ID3D12Device5* device5;
			ctx.device->QueryInterface<ID3D12Device5>(&device5);

			device5->GetRaytracingAccelerationStructurePrebuildInfo(&blas_desc.Inputs, &as_brebuild_info);

			ID3D12Resource* scratch_buffer = create_buffer(ctx.device, as_brebuild_info.ScratchDataSizeInBytes, D3D12_HEAP_TYPE_DEFAULT);
			ID3D12Resource* blas_buffer = create_buffer(ctx.device, as_brebuild_info.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT);

			blas_desc.ScratchAccelerationStructureData = scratch_buffer->GetGPUVirtualAddress();
			blas_desc.DestAccelerationStructureData = blas_buffer->GetGPUVirtualAddress();

			ID3D12GraphicsCommandList4* command_list4;
			ctx.command_list->QueryInterface<ID3D12GraphicsCommandList4>(&command_list4);

			ctx.command_allocator->Reset();
			command_list4->Reset(ctx.command_allocator, nullptr);

			command_list4->BuildRaytracingAccelerationStructure(&blas_desc, 0, nullptr);

			command_list4->Close();

			ctx.queue->ExecuteCommandLists(1, reinterpret_cast<ID3D12CommandList**>(&ctx.command_list));

			ctx.queue->Signal(ctx.fence, ++ctx.fence_value);
			ctx.fence->SetEventOnCompletion(ctx.fence_value, ctx.fence_event);

			WaitForSingleObject(ctx.fence_event, INFINITE);

			scratch_buffer->Release();

			return blas_buffer;
		}

		bool LoadMeshFromGLTF(Context& ctx, DescriptorHeap& descriptor_heap, const char* filename)
		{
			tinygltf::Model model;
			tinygltf::TinyGLTF loader;
			std::string err;
			std::string warn;

			loader.LoadBinaryFromFile(&model, &err, &warn, filename);

			auto mesh = model.meshes[0];
			auto primitive = mesh.primitives[0];
			auto attributes = primitive.attributes;
			auto indices_id = primitive.indices;
			auto material_id = primitive.material;

			auto material = model.materials[material_id];


			std::vector<uint16_t> indices = LoadDataFromGLTFAccessor<uint16_t>(model, indices_id);
			std::vector<float> positions = LoadDataFromGLTFAccessor<float>(model, attributes["POSITION"]);
			std::vector<float> normals = LoadDataFromGLTFAccessor<float>(model, attributes["NORMAL"]);
			std::vector<float> uvs = LoadDataFromGLTFAccessor<float>(model, attributes["TEXCOORD_0"]);

			ID3D12Resource* indices_buffer = create_buffer_from_data(ctx, indices.data(), indices.size() * sizeof(uint16_t));
			ID3D12Resource* positions_buffer = create_buffer_from_data(ctx, positions.data(), positions.size() * sizeof(float));
			ID3D12Resource* normals_buffer = create_buffer_from_data(ctx, normals.data(), normals.size() * sizeof(float));
			ID3D12Resource* uvs_buffer = create_buffer_from_data(ctx, uvs.data(), uvs.size() * sizeof(float));

			auto material_texture = material.pbrMetallicRoughness.baseColorTexture;

			auto texture = model.textures[material_texture.index];

			auto image = model.images[texture.source];

			ID3D12Resource* albedo_texture = create_texture2d_from_data(ctx, image.width, image.height, image.image);

			material_texture = material.emissiveTexture;

			texture = model.textures[material_texture.index];

			image = model.images[texture.source];

			ID3D12Resource* emissive_texture = create_texture2d_from_data(ctx, image.width, image.height, image.image);

			D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
			srv_desc.Format = DXGI_FORMAT_R16_UINT;
			srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
			srv_desc.Buffer.NumElements = indices.size();
			srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
			ctx.device->CreateShaderResourceView(indices_buffer, &srv_desc, descriptor_heap.GetCPUHandle(descriptor_heap_base + 0));

			srv_desc.Format = DXGI_FORMAT_R32G32B32_FLOAT;
			srv_desc.Buffer.NumElements = positions.size() / 3;
			ctx.device->CreateShaderResourceView(positions_buffer, &srv_desc, descriptor_heap.GetCPUHandle(descriptor_heap_base + 1));

			ctx.device->CreateShaderResourceView(normals_buffer, &srv_desc, descriptor_heap.GetCPUHandle(descriptor_heap_base + 2));

			srv_desc.Format = DXGI_FORMAT_R32G32_FLOAT;
			srv_desc.Buffer.NumElements = uvs.size() / 2;

			ctx.device->CreateShaderResourceView(uvs_buffer, &srv_desc, descriptor_heap.GetCPUHandle(descriptor_heap_base + 3));

			ctx.device->CreateShaderResourceView(albedo_texture, nullptr, descriptor_heap.GetCPUHandle(descriptor_heap_base + 4));
			ctx.device->CreateShaderResourceView(emissive_texture, nullptr, descriptor_heap.GetCPUHandle(descriptor_heap_base + 5));

			number_of_vertices = indices.size();

			blas = create_blas(ctx, indices_buffer, indices.size(), positions_buffer, positions.size() / 3);

			return true;

		}

		DirectX::XMMATRIX location;
		DirectX::XMMATRIX rotation;
		DirectX::XMMATRIX scaling;

		size_t descriptor_heap_base;
		uint32_t number_of_vertices;
		ID3D12Resource* blas;
	};
}
