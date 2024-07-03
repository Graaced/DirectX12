#include "aiv.h"

#include <thread>
#include <shared_mutex>


#include <math.h>
#define radians(degrees) ((degrees) * M_PI / 180.0)

std::shared_mutex render_lock;

void presenter(IDXGIFactory2* factory2, ID3D12Device2* device, ID3D12Resource* target)
{
	aiv::Context ctx = {};

	ctx.device = device;

	D3D12_COMMAND_QUEUE_DESC queue_desc = {};
	queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
	COMCHECK(ctx.device->CreateCommandQueue(&queue_desc, COMPTR(ctx.queue)));

	COMCHECK(ctx.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, COMPTR(ctx.fence)));

	ctx.fence_event = CreateEvent(NULL, FALSE, FALSE, NULL);

	COMCHECK(ctx.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, COMPTR(ctx.command_allocator)));

	COMCHECK(ctx.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, ctx.command_allocator, nullptr, COMPTR(ctx.command_list)));

	ctx.command_list->Close();

	SDL_Window* window = SDL_CreateWindow("DirectX12", 100, 100, 512, 512, 0);

	SDL_SysWMinfo wm_info;
	SDL_VERSION(&wm_info.version);

	SDL_GetWindowWMInfo(window, &wm_info);

	DXGI_SWAP_CHAIN_DESC1 swap_chain_desc1 = {};
	swap_chain_desc1.Width = 512;
	swap_chain_desc1.Height = 512;
	swap_chain_desc1.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
	swap_chain_desc1.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swap_chain_desc1.BufferCount = 3;
	swap_chain_desc1.Scaling = DXGI_SCALING_STRETCH;
	swap_chain_desc1.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	swap_chain_desc1.SampleDesc.Count = 1;
	swap_chain_desc1.AlphaMode = DXGI_ALPHA_MODE_IGNORE;

	IDXGISwapChain3* swap_chain = nullptr;
	COMCHECK(factory2->CreateSwapChainForHwnd(ctx.queue, wm_info.info.win.window, &swap_chain_desc1, nullptr, nullptr, (IDXGISwapChain1**)&swap_chain));

	for (;;)
	{
		SDL_Event _event;
		while (SDL_PollEvent(&_event))
		{

		}

		ID3D12Resource* current_back_buffer = nullptr;
		swap_chain->GetBuffer(swap_chain->GetCurrentBackBufferIndex(), COMPTR(current_back_buffer));

		render_lock.lock_shared();

		aiv::copy_resource(ctx, target, current_back_buffer,
			D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);

		render_lock.unlock_shared();

		swap_chain->Present(1, 0);
	}
}

int main(int argc, char** argv)
{

	aiv::Context ctx = {};

	IDXGIFactory* factory = nullptr;
	COMCHECK(CreateDXGIFactory2(DXGI_CREATE_FACTORY_DEBUG, COMPTR(factory)));

	IDXGIFactory2* factory2 = nullptr;
	COMCHECK(factory->QueryInterface<IDXGIFactory2>(&factory2));

	UINT adapter_index = 0;
	IDXGIAdapter1* adapter = nullptr;
	std::vector<IDXGIAdapter1*> adapters;
	for (;;)
	{
		adapter = nullptr;
		if (factory2->EnumAdapters1(adapter_index, &adapter) != S_OK)
		{
			break;
		}

		adapters.push_back(adapter);

		adapter_index++;
	}

	for (IDXGIAdapter1* current_adapter : adapters)
	{
		DXGI_ADAPTER_DESC1 adapter_desc;
		current_adapter->GetDesc1(&adapter_desc);
		std::wcout << adapter_desc.Description << " " << adapter_desc.DedicatedVideoMemory / 1024 / 1024 << std::endl;
	}

	IDXGIAdapter1* best_adapter = adapters[1];

	ID3D12Debug* debug = nullptr;
	D3D12GetDebugInterface(COMPTR(debug));
	debug->EnableDebugLayer();

	COMCHECK(D3D12CreateDevice(best_adapter, D3D_FEATURE_LEVEL_12_0, COMPTR(ctx.device)));

	std::cout << "Number of GPU nodes: " << ctx.device->GetNodeCount() << std::endl;

	D3D12_COMMAND_QUEUE_DESC queue_desc = {};
	queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
	COMCHECK(ctx.device->CreateCommandQueue(&queue_desc, COMPTR(ctx.queue)));

	COMCHECK(ctx.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, COMPTR(ctx.fence)));

	ctx.fence_event = CreateEvent(NULL, FALSE, FALSE, NULL);

	COMCHECK(ctx.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, COMPTR(ctx.command_allocator)));

	COMCHECK(ctx.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, ctx.command_allocator, nullptr, COMPTR(ctx.command_list)));

	ctx.command_list->Close();

	std::vector<D3D12_DESCRIPTOR_RANGE1> ranges;

	D3D12_DESCRIPTOR_RANGE1 range0 = {};
	range0.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
	range0.NumDescriptors = 7; // t0 - t6
	range0.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

	ranges.push_back(range0);

	std::vector<D3D12_DESCRIPTOR_RANGE1> ranges2;
	D3D12_DESCRIPTOR_RANGE1 range1 = {};
	range1.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
	range1.NumDescriptors = 1; // s0
	range1.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

	ranges2.push_back(range1);

	std::vector<D3D12_ROOT_PARAMETER1> params;

	D3D12_ROOT_PARAMETER1 param0 = {};
	param0.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
	param0.DescriptorTable.NumDescriptorRanges = ranges.size();
	param0.DescriptorTable.pDescriptorRanges = ranges.data();
	params.push_back(param0);

	D3D12_ROOT_PARAMETER1 param1 = {};
	param1.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
	param1.Constants.Num32BitValues = 16 + 16 + 16 + 1;
	params.push_back(param1);

	D3D12_ROOT_PARAMETER1 param2 = {};
	param2.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
	param2.DescriptorTable.NumDescriptorRanges = ranges2.size();
	param2.DescriptorTable.pDescriptorRanges = ranges2.data();
	params.push_back(param2);

	D3D12_VERSIONED_ROOT_SIGNATURE_DESC versioned_root_desc = {};
	versioned_root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
	versioned_root_desc.Desc_1_1.NumParameters = params.size();
	versioned_root_desc.Desc_1_1.pParameters = params.data();

	ID3DBlob* serialized_root_signature;
	COMCHECK(D3D12SerializeVersionedRootSignature(&versioned_root_desc, &serialized_root_signature, nullptr));

	ID3D12RootSignature* root_signature;
	COMCHECK(ctx.device->CreateRootSignature(0,
		serialized_root_signature->GetBufferPointer(),
		serialized_root_signature->GetBufferSize(), COMPTR(root_signature)));


	aiv::DescriptorHeap descriptor_heap_cbv_srv_uav(ctx.device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true);
	aiv::DescriptorHeap descriptor_heap_rtv(ctx.device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, false);
	aiv::DescriptorHeap descriptor_heap_dsv(ctx.device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, false);
	aiv::DescriptorHeap descriptor_heap_sampler(ctx.device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, true);

	ID3D12Resource* target = aiv::create_texture2d(ctx.device, 512, 512, DXGI_FORMAT_B8G8R8A8_UNORM, L"Texture Target0");
	ctx.device->CreateRenderTargetView(target, nullptr, descriptor_heap_rtv.GetCPUHandle(0));

	ID3D12Resource* target2 = aiv::create_texture2d(ctx.device, 512, 512, DXGI_FORMAT_B8G8R8A8_UNORM, L"Texture Target1");
	ctx.device->CreateRenderTargetView(target2, nullptr, descriptor_heap_rtv.GetCPUHandle(1));

	ID3D12Resource* depth = aiv::create_depth(ctx.device, 512, 512, L"Depth");
	ctx.device->CreateDepthStencilView(depth, nullptr, descriptor_heap_dsv.GetCPUHandle(0));

	D3D12_SAMPLER_DESC sampler_desc = {};
	sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
	sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
	sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
	sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
	ctx.device->CreateSampler(&sampler_desc, descriptor_heap_sampler.GetCPUHandle(0));


	ID3D12PipelineState* rasterizer_pipeline;

	ID3D12Device2* device2;
	COMCHECK(ctx.device->QueryInterface<ID3D12Device2>(&device2));

	std::vector<char> pipeline_stream;
	aiv::append_subobject_to_stream(pipeline_stream, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root_signature);

	D3D12_SHADER_BYTECODE vs;
	std::vector<char> vs_code = aiv::load_file("C:/Users/giuli/source/repos/3o anno/DirectX12/DirectXAIV/vertex.bin");
	vs.pShaderBytecode = vs_code.data();
	vs.BytecodeLength = vs_code.size();
	aiv::append_subobject_to_stream(pipeline_stream, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, vs);

	D3D12_SHADER_BYTECODE ps;
	std::vector<char> ps_code = aiv::load_file("C:/Users/giuli/source/repos/3o anno/DirectX12/DirectXAIV/pixel.bin");
	ps.pShaderBytecode = ps_code.data();
	ps.BytecodeLength = ps_code.size();
	aiv::append_subobject_to_stream(pipeline_stream, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps);

	// topology
	aiv::append_subobject_to_stream(pipeline_stream, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);

	// render target formats
	D3D12_RT_FORMAT_ARRAY rts = {};
	rts.NumRenderTargets = 2;
	rts.RTFormats[0] = DXGI_FORMAT_B8G8R8A8_UNORM;
	rts.RTFormats[1] = DXGI_FORMAT_B8G8R8A8_UNORM;
	aiv::append_subobject_to_stream(pipeline_stream, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, rts);

	aiv::append_subobject_to_stream(pipeline_stream, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, DXGI_FORMAT_D32_FLOAT);

	D3D12_RASTERIZER_DESC rasterizer_desc = {};
	rasterizer_desc.FillMode = D3D12_FILL_MODE_SOLID;
	rasterizer_desc.CullMode = D3D12_CULL_MODE_NONE;
	rasterizer_desc.FrontCounterClockwise = true;

	aiv::append_subobject_to_stream(pipeline_stream, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, rasterizer_desc);

	D3D12_PIPELINE_STATE_STREAM_DESC stream_desc = {};
	stream_desc.pPipelineStateSubobjectStream = pipeline_stream.data();
	stream_desc.SizeInBytes = pipeline_stream.size();


	COMCHECK(device2->CreatePipelineState(&stream_desc, COMPTR(rasterizer_pipeline)));

	SDL_Init(SDL_INIT_VIDEO);

	std::thread thread0(presenter, factory2, device2, target);
	std::thread thread1(presenter, factory2, device2, target2);

	float color = 0.5;

	float rotation = 0;

	float camera_z = 10;

	std::vector<std::shared_ptr<aiv::Actor>> actors;

	std::shared_ptr<aiv::Actor> actor0 = std::make_shared<aiv::Actor>();
	actor0->LoadMeshFromGLTF(ctx, descriptor_heap_cbv_srv_uav, "E:\Lezioni Programmazione\LEZIONI 3 ANNO\DirectX12\DamagedHelmet.glb");
	actor0->rotation = DirectX::XMMatrixRotationX(radians(90));
	actors.push_back(actor0);

	std::shared_ptr<aiv::Actor> actor1 = std::make_shared<aiv::Actor>();
	actor1->LoadMeshFromGLTF(ctx, descriptor_heap_cbv_srv_uav, "E:\Lezioni Programmazione\LEZIONI 3 ANNO\DirectX12\DamagedHelmet.glb");
	actor1->rotation = DirectX::XMMatrixRotationX(radians(90));
	actor1->location = DirectX::XMMatrixTranslation(0, 0, -10);
	actor1->scaling = DirectX::XMMatrixScaling(10, 10, 10);
	actors.push_back(actor1);

	int previous_mouse_x, previous_mouse_y = 0;

	DirectX::XMMATRIX camera_translation = DirectX::XMMatrixTranslation(0, 0, 4);
	DirectX::XMMATRIX camera_rotation = DirectX::XMMatrixIdentity();

	D3D12_RAYTRACING_INSTANCE_DESC instance_descs[2] = {};
	instance_descs[0].AccelerationStructure = actor0->blas->GetGPUVirtualAddress();
	instance_descs[0].InstanceID = 100;
	instance_descs[0].InstanceMask = 0xff;
	instance_descs[0].Flags = D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE;
	DirectX::XMMATRIX world = actor0->scaling * actor0->rotation * actor0->location;
	::memcpy(instance_descs[0].Transform, DirectX::XMMatrixTranspose(world).r, sizeof(float) * 12);

	instance_descs[1].AccelerationStructure = actor1->blas->GetGPUVirtualAddress();
	instance_descs[1].InstanceID = 200;
	instance_descs[1].InstanceMask = 0xff;
	instance_descs[1].Flags = D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE;
	world = actor1->scaling * actor1->rotation * actor1->location;
	::memcpy(instance_descs[1].Transform, DirectX::XMMatrixTranspose(world).r, sizeof(float) * 12);

	ID3D12Resource* instances_buffer = aiv::create_buffer_from_data(ctx, instance_descs, sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * 2);

	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlas_desc = {};
	tlas_desc.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
	tlas_desc.Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
	tlas_desc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
	tlas_desc.Inputs.NumDescs = 2;
	tlas_desc.Inputs.InstanceDescs = instances_buffer->GetGPUVirtualAddress();

	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO as_brebuild_info = {};

	ID3D12Device5* device5;
	ctx.device->QueryInterface<ID3D12Device5>(&device5);

	device5->GetRaytracingAccelerationStructurePrebuildInfo(&tlas_desc.Inputs, &as_brebuild_info);

	auto scratch_buffer = aiv::create_buffer(ctx.device, as_brebuild_info.ScratchDataSizeInBytes, D3D12_HEAP_TYPE_DEFAULT);
	auto tlas_buffer = aiv::create_buffer(ctx.device, as_brebuild_info.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT);
	auto scratch_update_buffer = aiv::create_buffer(ctx.device, as_brebuild_info.UpdateScratchDataSizeInBytes, D3D12_HEAP_TYPE_DEFAULT);

	tlas_desc.ScratchAccelerationStructureData = scratch_buffer->GetGPUVirtualAddress();
	tlas_desc.DestAccelerationStructureData = tlas_buffer->GetGPUVirtualAddress();

	ID3D12GraphicsCommandList4* command_list4;
	ctx.command_list->QueryInterface<ID3D12GraphicsCommandList4>(&command_list4);

	ctx.command_allocator->Reset();
	command_list4->Reset(ctx.command_allocator, nullptr);

	command_list4->BuildRaytracingAccelerationStructure(&tlas_desc, 0, nullptr);

	command_list4->Close();

	ctx.queue->ExecuteCommandLists(1, reinterpret_cast<ID3D12CommandList**>(&ctx.command_list));

	ctx.queue->Signal(ctx.fence, ++ctx.fence_value);
	ctx.fence->SetEventOnCompletion(ctx.fence_value, ctx.fence_event);

	WaitForSingleObject(ctx.fence_event, INFINITE);

	scratch_buffer->Release();

	D3D12_SHADER_RESOURCE_VIEW_DESC tlas_view = {};
	tlas_view.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
	tlas_view.RaytracingAccelerationStructure.Location = tlas_buffer->GetGPUVirtualAddress();
	tlas_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

	ctx.device->CreateShaderResourceView(nullptr, &tlas_view, descriptor_heap_cbv_srv_uav.GetCPUHandle(6));
	ctx.device->CreateShaderResourceView(nullptr, &tlas_view, descriptor_heap_cbv_srv_uav.GetCPUHandle(14));

	ID3D12Resource* tlas_update_buffer = aiv::create_buffer(ctx.device, sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * 2, D3D12_HEAP_TYPE_UPLOAD);

	for (;;)
	{

		int mouse_x, mouse_y = 0;
		uint32_t mouse_mask = SDL_GetMouseState(&mouse_x, &mouse_y);

		if (mouse_mask)
		{
			if (mouse_x > previous_mouse_x)
			{
				camera_rotation *= DirectX::XMMatrixRotationY(-0.01);
			}

			else if (mouse_x < previous_mouse_x)
			{
				camera_rotation *= DirectX::XMMatrixRotationY(0.01);
			}
		}

		previous_mouse_x = mouse_x;
		previous_mouse_y = mouse_y;

		const uint8_t* keys = SDL_GetKeyboardState(nullptr);

		if (keys[SDL_SCANCODE_W])
		{
			camera_translation *= DirectX::XMMatrixTranslationFromVector(DirectX::XMVector3Transform(DirectX::XMVECTORF32{ 0, 0, -0.1, 0 }, camera_rotation));
		}

		if (keys[SDL_SCANCODE_S])
		{
			camera_translation *= DirectX::XMMatrixTranslationFromVector(DirectX::XMVector3Transform(DirectX::XMVECTORF32{ 0, 0, 0.1, 0 }, camera_rotation));
		}

		if (keys[SDL_SCANCODE_D])
		{
			camera_translation *= DirectX::XMMatrixTranslationFromVector(DirectX::XMVector3Transform(DirectX::XMVECTORF32{ 0.1, 0, 0, 0 }, camera_rotation));
		}

		if (keys[SDL_SCANCODE_A])
		{
			camera_translation *= DirectX::XMMatrixTranslationFromVector(DirectX::XMVector3Transform(DirectX::XMVECTORF32{ -0.1, 0, 0, 0 }, camera_rotation));
		}

		if (keys[SDL_SCANCODE_E])
		{
			camera_translation *= DirectX::XMMatrixTranslationFromVector(DirectX::XMVector3Transform(DirectX::XMVECTORF32{ 0, 0.1, 0, 0 }, camera_rotation));
		}

		if (keys[SDL_SCANCODE_Q])
		{
			camera_translation *= DirectX::XMMatrixTranslationFromVector(DirectX::XMVector3Transform(DirectX::XMVECTORF32{ 0, -0.1, 0, 0 }, camera_rotation));
		}

		DirectX::XMMATRIX camera = camera_rotation * camera_translation;

		DirectX::XMMATRIX view = DirectX::XMMatrixInverse(nullptr, camera);

		DirectX::XMMATRIX projection = DirectX::XMMatrixPerspectiveFovRH(radians(60), 1.0, 0.01, 1000);

		ctx.command_allocator->Reset();
		ctx.command_list->Reset(ctx.command_allocator, nullptr);

		color += 0.01;
		if (color > 1)
		{
			color = 0.5;
		}

		actor0->rotation *= DirectX::XMMatrixRotationY(0.01);

		world = actor0->scaling * actor0->rotation * actor0->location;
		::memcpy(instance_descs[0].Transform, DirectX::XMMatrixTranspose(world).r, sizeof(float) * 12);
		world = actor1->scaling * actor1->rotation * actor1->location;
		::memcpy(instance_descs[1].Transform, DirectX::XMMatrixTranspose(world).r, sizeof(float) * 12);

		void* mapped_tlas_update;
		tlas_update_buffer->Map(0, nullptr, &mapped_tlas_update);

		::memcpy(mapped_tlas_update, instance_descs, sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * 2);

		tlas_update_buffer->Unmap(0, nullptr);

		ctx.command_list->CopyResource(instances_buffer, tlas_update_buffer);

		ctx.command_list->Close();

		ctx.queue->ExecuteCommandLists(1, reinterpret_cast<ID3D12CommandList**>(&ctx.command_list));

		ctx.queue->Signal(ctx.fence, ++ctx.fence_value);
		ctx.fence->SetEventOnCompletion(ctx.fence_value, ctx.fence_event);

		WaitForSingleObject(ctx.fence_event, INFINITE);

		ctx.command_allocator->Reset();
		ctx.command_list->Reset(ctx.command_allocator, nullptr);

		tlas_desc.ScratchAccelerationStructureData = scratch_update_buffer->GetGPUVirtualAddress();
		tlas_desc.DestAccelerationStructureData = tlas_buffer->GetGPUVirtualAddress();
		tlas_desc.SourceAccelerationStructureData = tlas_buffer->GetGPUVirtualAddress();
		tlas_desc.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;

		command_list4->BuildRaytracingAccelerationStructure(&tlas_desc, 0, nullptr);

		ctx.command_list->Close();

		ctx.queue->ExecuteCommandLists(1, reinterpret_cast<ID3D12CommandList**>(&ctx.command_list));

		ctx.queue->Signal(ctx.fence, ++ctx.fence_value);
		ctx.fence->SetEventOnCompletion(ctx.fence_value, ctx.fence_event);

		WaitForSingleObject(ctx.fence_event, INFINITE);

		ctx.command_allocator->Reset();
		ctx.command_list->Reset(ctx.command_allocator, nullptr);

		ctx.command_list->SetPipelineState(rasterizer_pipeline);
		ctx.command_list->SetGraphicsRootSignature(root_signature);
		ID3D12DescriptorHeap* descriptor_heaps[] = { descriptor_heap_cbv_srv_uav.GetComPtr(), descriptor_heap_sampler.GetComPtr() };
		ctx.command_list->SetDescriptorHeaps(2, descriptor_heaps);

		ctx.command_list->SetGraphicsRootDescriptorTable(2, descriptor_heap_sampler.GetGPUHandle(0));

		ctx.command_list->SetGraphicsRoot32BitConstants(1, 16, view.r, 16);
		ctx.command_list->SetGraphicsRoot32BitConstants(1, 16, projection.r, 16 * 2);
		ctx.command_list->SetGraphicsRoot32BitConstants(1, 1, &color, 16 * 3);

		ctx.command_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

		float black[] = { 0, 0, 0, 0 };
		ctx.command_list->ClearRenderTargetView(descriptor_heap_rtv.GetCPUHandle(0), black, 0, nullptr);
		ctx.command_list->ClearRenderTargetView(descriptor_heap_rtv.GetCPUHandle(1), black, 0, nullptr);

		ctx.command_list->ClearDepthStencilView(descriptor_heap_dsv.GetCPUHandle(0), D3D12_CLEAR_FLAG_DEPTH, 1, 0, 0, nullptr);

		D3D12_CPU_DESCRIPTOR_HANDLE rtvs[] = { descriptor_heap_rtv.GetCPUHandle(0),descriptor_heap_rtv.GetCPUHandle(1) };
		D3D12_CPU_DESCRIPTOR_HANDLE dsv = descriptor_heap_dsv.GetCPUHandle(0);
		ctx.command_list->OMSetRenderTargets(2, rtvs, true, &dsv);

		D3D12_VIEWPORT viewport = {};
		viewport.Width = 512;
		viewport.Height = 512;
		viewport.MaxDepth = 1;
		ctx.command_list->RSSetViewports(1, &viewport);

		D3D12_RECT scissor = {};
		scissor.right = 512;
		scissor.bottom = 512;
		ctx.command_list->RSSetScissorRects(1, &scissor);

		for (std::shared_ptr<aiv::Actor> actor : actors)
		{
			ctx.command_list->SetGraphicsRootDescriptorTable(0, descriptor_heap_cbv_srv_uav.GetGPUHandle(actor->descriptor_heap_base));
			world = actor->scaling * actor->rotation * actor->location;
			ctx.command_list->SetGraphicsRoot32BitConstants(1, 16, world.r, 0);
			ctx.command_list->DrawInstanced(actor->number_of_vertices, 1, 0, 0);
		}
		ctx.command_list->Close();

		render_lock.lock();

		ctx.queue->ExecuteCommandLists(1, reinterpret_cast<ID3D12CommandList**>(&ctx.command_list));

		ctx.queue->Signal(ctx.fence, ++ctx.fence_value);
		ctx.fence->SetEventOnCompletion(ctx.fence_value, ctx.fence_event);

		WaitForSingleObject(ctx.fence_event, INFINITE);

		render_lock.unlock();

	}


	return 0;
}
