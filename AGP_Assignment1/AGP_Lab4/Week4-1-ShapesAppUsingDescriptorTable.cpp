//***************************************************************************************
// ShapesApp.cpp 
//
// Hold down '1' key to view scene in wireframe mode.
//***************************************************************************************

#include "../Common/d3dApp.h"
#include "../Common/MathHelper.h"
#include "../Common/UploadBuffer.h"
#include "../Common/GeometryGenerator.h"
#include "FrameResource.h"

using Microsoft::WRL::ComPtr;
using namespace DirectX;
using namespace DirectX::PackedVector;

const int gNumFrameResources = 3;

// Lightweight structure stores parameters to draw a shape.  This will
// vary from app-to-app.
struct RenderItem
{
	RenderItem() = default;

    // World matrix of the shape that describes the object's local space
    // relative to the world space, which defines the position, orientation,
    // and scale of the object in the world.
    XMFLOAT4X4 World = MathHelper::Identity4x4();

	// Dirty flag indicating the object data has changed and we need to update the constant buffer.
	// Because we have an object cbuffer for each FrameResource, we have to apply the
	// update to each FrameResource.  Thus, when we modify obect data we should set 
	// NumFramesDirty = gNumFrameResources so that each frame resource gets the update.
	int NumFramesDirty = gNumFrameResources;

	// Index into GPU constant buffer corresponding to the ObjectCB for this render item.
	UINT ObjCBIndex = -1;

	MeshGeometry* Geo = nullptr;

    // Primitive topology.
    D3D12_PRIMITIVE_TOPOLOGY PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

    // DrawIndexedInstanced parameters.
    UINT IndexCount = 0;
    UINT StartIndexLocation = 0;
    int BaseVertexLocation = 0;
};

class ShapesApp : public D3DApp
{
public:
    ShapesApp(HINSTANCE hInstance);
    ShapesApp(const ShapesApp& rhs) = delete;
    ShapesApp& operator=(const ShapesApp& rhs) = delete;
    ~ShapesApp();

    virtual bool Initialize()override;

private:
    virtual void OnResize()override;
    virtual void Update(const GameTimer& gt)override;
    virtual void Draw(const GameTimer& gt)override;

    virtual void OnMouseDown(WPARAM btnState, int x, int y)override;
    virtual void OnMouseUp(WPARAM btnState, int x, int y)override;
    virtual void OnMouseMove(WPARAM btnState, int x, int y)override;

    void OnKeyboardInput(const GameTimer& gt);
	void UpdateCamera(const GameTimer& gt);
	void UpdateObjectCBs(const GameTimer& gt);
	void UpdateMainPassCB(const GameTimer& gt);

    void BuildDescriptorHeaps();
    void BuildConstantBufferViews();
    void BuildRootSignature();
    void BuildShadersAndInputLayout();
    void BuildShapeGeometry();
    void BuildPSOs();
    void BuildFrameResources();
    void BuildRenderItems();
    void DrawRenderItems(ID3D12GraphicsCommandList* cmdList, const std::vector<RenderItem*>& ritems);
 
private:

    std::vector<std::unique_ptr<FrameResource>> mFrameResources;
    FrameResource* mCurrFrameResource = nullptr;
    int mCurrFrameResourceIndex = 0;

    ComPtr<ID3D12RootSignature> mRootSignature = nullptr;
    ComPtr<ID3D12DescriptorHeap> mCbvHeap = nullptr;

	ComPtr<ID3D12DescriptorHeap> mSrvDescriptorHeap = nullptr;

	std::unordered_map<std::string, std::unique_ptr<MeshGeometry>> mGeometries;
	std::unordered_map<std::string, ComPtr<ID3DBlob>> mShaders;
    std::unordered_map<std::string, ComPtr<ID3D12PipelineState>> mPSOs;

    std::vector<D3D12_INPUT_ELEMENT_DESC> mInputLayout;

	// List of all the render items.
	std::vector<std::unique_ptr<RenderItem>> mAllRitems;

	// Render items divided by PSO.
	std::vector<RenderItem*> mOpaqueRitems;

    PassConstants mMainPassCB;

    UINT mPassCbvOffset = 0;

    bool mIsWireframe = false;

	XMFLOAT3 mEyePos = { 0.0f, 0.0f, 0.0f };
	XMFLOAT4X4 mView = MathHelper::Identity4x4();
	XMFLOAT4X4 mProj = MathHelper::Identity4x4();

    float mTheta = 1.5f*XM_PI;
    float mPhi = 0.2f*XM_PI;
    float mRadius = 15.0f;

    POINT mLastMousePos;
};

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevInstance,
    PSTR cmdLine, int showCmd)
{
    // Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

    try
    {
        ShapesApp theApp(hInstance);
        if(!theApp.Initialize())
            return 0;

        return theApp.Run();
    }
    catch(DxException& e)
    {
        MessageBox(nullptr, e.ToString().c_str(), L"HR Failed", MB_OK);
        return 0;
    }
}

ShapesApp::ShapesApp(HINSTANCE hInstance)
    : D3DApp(hInstance)
{
}

ShapesApp::~ShapesApp()
{
    if(md3dDevice != nullptr)
        FlushCommandQueue();
}

bool ShapesApp::Initialize()
{
    if(!D3DApp::Initialize())
        return false;

    // Reset the command list to prep for initialization commands.
    ThrowIfFailed(mCommandList->Reset(mDirectCmdListAlloc.Get(), nullptr));

    BuildRootSignature();
    BuildShadersAndInputLayout();
    BuildShapeGeometry();
    BuildRenderItems();
    BuildFrameResources();
    BuildDescriptorHeaps();
    BuildConstantBufferViews();
    BuildPSOs();

    // Execute the initialization commands.
    ThrowIfFailed(mCommandList->Close());
    ID3D12CommandList* cmdsLists[] = { mCommandList.Get() };
    mCommandQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);

    // Wait until initialization is complete.
    FlushCommandQueue();

    return true;
}
 
void ShapesApp::OnResize()
{
    D3DApp::OnResize();

    // The window resized, so update the aspect ratio and recompute the projection matrix.
    XMMATRIX P = XMMatrixPerspectiveFovLH(0.25f*MathHelper::Pi, AspectRatio(), 1.0f, 1000.0f);
    XMStoreFloat4x4(&mProj, P);
}

void ShapesApp::Update(const GameTimer& gt)
{
    OnKeyboardInput(gt);
	UpdateCamera(gt);

    // Cycle through the circular frame resource array.
    mCurrFrameResourceIndex = (mCurrFrameResourceIndex + 1) % gNumFrameResources;
    mCurrFrameResource = mFrameResources[mCurrFrameResourceIndex].get();

    // Has the GPU finished processing the commands of the current frame resource?
    // If not, wait until the GPU has completed commands up to this fence point.
    if(mCurrFrameResource->Fence != 0 && mFence->GetCompletedValue() < mCurrFrameResource->Fence)
    {
        HANDLE eventHandle = CreateEventEx(nullptr, nullptr, false, EVENT_ALL_ACCESS);
        ThrowIfFailed(mFence->SetEventOnCompletion(mCurrFrameResource->Fence, eventHandle));
        WaitForSingleObject(eventHandle, INFINITE);
        CloseHandle(eventHandle);
    }

	UpdateObjectCBs(gt);
	UpdateMainPassCB(gt);
}

void ShapesApp::Draw(const GameTimer& gt)
{
    auto cmdListAlloc = mCurrFrameResource->CmdListAlloc;

    // Reuse the memory associated with command recording.
    // We can only reset when the associated command lists have finished execution on the GPU.
    ThrowIfFailed(cmdListAlloc->Reset());

    // A command list can be reset after it has been added to the command queue via ExecuteCommandList.
    // Reusing the command list reuses memory.
    if(mIsWireframe)
    {
        ThrowIfFailed(mCommandList->Reset(cmdListAlloc.Get(), mPSOs["opaque_wireframe"].Get()));
    }
    else
    {
        ThrowIfFailed(mCommandList->Reset(cmdListAlloc.Get(), mPSOs["opaque"].Get()));
    }

    mCommandList->RSSetViewports(1, &mScreenViewport);
    mCommandList->RSSetScissorRects(1, &mScissorRect);

    // Indicate a state transition on the resource usage.
	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(),
		D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));

    // Clear the back buffer and depth buffer.
    mCommandList->ClearRenderTargetView(CurrentBackBufferView(), Colors::LightSteelBlue, 0, nullptr);
    mCommandList->ClearDepthStencilView(DepthStencilView(), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, nullptr);

    // Specify the buffers we are going to render to.
    mCommandList->OMSetRenderTargets(1, &CurrentBackBufferView(), true, &DepthStencilView());

    ID3D12DescriptorHeap* descriptorHeaps[] = { mCbvHeap.Get() };
    mCommandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

	mCommandList->SetGraphicsRootSignature(mRootSignature.Get());

    int passCbvIndex = mPassCbvOffset + mCurrFrameResourceIndex;
    auto passCbvHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(mCbvHeap->GetGPUDescriptorHandleForHeapStart());
    passCbvHandle.Offset(passCbvIndex, mCbvSrvUavDescriptorSize);
    mCommandList->SetGraphicsRootDescriptorTable(1, passCbvHandle);

    DrawRenderItems(mCommandList.Get(), mOpaqueRitems);

    // Indicate a state transition on the resource usage.
	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(),
		D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));

    // Done recording commands.
    ThrowIfFailed(mCommandList->Close());

    // Add the command list to the queue for execution.
    ID3D12CommandList* cmdsLists[] = { mCommandList.Get() };
    mCommandQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);

    // Swap the back and front buffers
    ThrowIfFailed(mSwapChain->Present(0, 0));
	mCurrBackBuffer = (mCurrBackBuffer + 1) % SwapChainBufferCount;

    // Advance the fence value to mark commands up to this fence point.
    mCurrFrameResource->Fence = ++mCurrentFence;
    
    // Add an instruction to the command queue to set a new fence point. 
    // Because we are on the GPU timeline, the new fence point won't be 
    // set until the GPU finishes processing all the commands prior to this Signal().
    mCommandQueue->Signal(mFence.Get(), mCurrentFence);
}

void ShapesApp::OnMouseDown(WPARAM btnState, int x, int y)
{
    mLastMousePos.x = x;
    mLastMousePos.y = y;

    SetCapture(mhMainWnd);
}

void ShapesApp::OnMouseUp(WPARAM btnState, int x, int y)
{
    ReleaseCapture();
}

void ShapesApp::OnMouseMove(WPARAM btnState, int x, int y)
{
    if((btnState & MK_LBUTTON) != 0)
    {
        // Make each pixel correspond to a quarter of a degree.
        float dx = XMConvertToRadians(0.25f*static_cast<float>(x - mLastMousePos.x));
        float dy = XMConvertToRadians(0.25f*static_cast<float>(y - mLastMousePos.y));

        // Update angles based on input to orbit camera around box.
        mTheta += dx;
        mPhi += dy;

        // Restrict the angle mPhi.
        mPhi = MathHelper::Clamp(mPhi, 0.1f, MathHelper::Pi - 0.1f);
    }
    else if((btnState & MK_RBUTTON) != 0)
    {
        // Make each pixel correspond to 0.2 unit in the scene.
        float dx = 0.05f*static_cast<float>(x - mLastMousePos.x);
        float dy = 0.05f*static_cast<float>(y - mLastMousePos.y);

        // Update the camera radius based on input.
        mRadius += dx - dy;

        // Restrict the radius.
        mRadius = MathHelper::Clamp(mRadius, 5.0f, 150.0f);
    }

    mLastMousePos.x = x;
    mLastMousePos.y = y;
}
 
void ShapesApp::OnKeyboardInput(const GameTimer& gt)
{
    if(GetAsyncKeyState('1') & 0x8000)
        mIsWireframe = true;
    else
        mIsWireframe = false;
}
 
void ShapesApp::UpdateCamera(const GameTimer& gt)
{
	// Convert Spherical to Cartesian coordinates.
	mEyePos.x = mRadius*sinf(mPhi)*cosf(mTheta);
	mEyePos.z = mRadius*sinf(mPhi)*sinf(mTheta);
	mEyePos.y = mRadius*cosf(mPhi);

	// Build the view matrix.
	XMVECTOR pos = XMVectorSet(mEyePos.x, mEyePos.y, mEyePos.z, 1.0f);
	XMVECTOR target = XMVectorZero();
	XMVECTOR up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

	XMMATRIX view = XMMatrixLookAtLH(pos, target, up);
	XMStoreFloat4x4(&mView, view);
}

void ShapesApp::UpdateObjectCBs(const GameTimer& gt)
{
	auto currObjectCB = mCurrFrameResource->ObjectCB.get();
	for(auto& e : mAllRitems)
	{
		// Only update the cbuffer data if the constants have changed.  
		// This needs to be tracked per frame resource.
		if(e->NumFramesDirty > 0)
		{
			XMMATRIX world = XMLoadFloat4x4(&e->World);

			ObjectConstants objConstants;
			XMStoreFloat4x4(&objConstants.World, XMMatrixTranspose(world));

			currObjectCB->CopyData(e->ObjCBIndex, objConstants);

			// Next FrameResource need to be updated too.
			e->NumFramesDirty--;
		}
	}
}

void ShapesApp::UpdateMainPassCB(const GameTimer& gt)
{
	XMMATRIX view = XMLoadFloat4x4(&mView);
	XMMATRIX proj = XMLoadFloat4x4(&mProj);

	XMMATRIX viewProj = XMMatrixMultiply(view, proj);
	XMMATRIX invView = XMMatrixInverse(&XMMatrixDeterminant(view), view);
	XMMATRIX invProj = XMMatrixInverse(&XMMatrixDeterminant(proj), proj);
	XMMATRIX invViewProj = XMMatrixInverse(&XMMatrixDeterminant(viewProj), viewProj);

	XMStoreFloat4x4(&mMainPassCB.View, XMMatrixTranspose(view));
	XMStoreFloat4x4(&mMainPassCB.InvView, XMMatrixTranspose(invView));
	XMStoreFloat4x4(&mMainPassCB.Proj, XMMatrixTranspose(proj));
	XMStoreFloat4x4(&mMainPassCB.InvProj, XMMatrixTranspose(invProj));
	XMStoreFloat4x4(&mMainPassCB.ViewProj, XMMatrixTranspose(viewProj));
	XMStoreFloat4x4(&mMainPassCB.InvViewProj, XMMatrixTranspose(invViewProj));
	mMainPassCB.EyePosW = mEyePos;
	mMainPassCB.RenderTargetSize = XMFLOAT2((float)mClientWidth, (float)mClientHeight);
	mMainPassCB.InvRenderTargetSize = XMFLOAT2(1.0f / mClientWidth, 1.0f / mClientHeight);
	mMainPassCB.NearZ = 1.0f;
	mMainPassCB.FarZ = 1000.0f;
	mMainPassCB.TotalTime = gt.TotalTime();
	mMainPassCB.DeltaTime = gt.DeltaTime();

	auto currPassCB = mCurrFrameResource->PassCB.get();
	currPassCB->CopyData(0, mMainPassCB);
}

//If we have 3 frame resources and n render items, then we have three 3n object constant
//buffers and 3 pass constant buffers.Hence we need 3(n + 1) constant buffer views(CBVs).
//Thus we will need to modify our CBV heap to include the additional descriptors :

void ShapesApp::BuildDescriptorHeaps()
{
    UINT objCount = (UINT)mOpaqueRitems.size();

    // Need a CBV descriptor for each object for each frame resource,
    // +1 for the perPass CBV for each frame resource.
    UINT numDescriptors = (objCount+1) * gNumFrameResources;

    // Save an offset to the start of the pass CBVs.  These are the last 3 descriptors.
    mPassCbvOffset = objCount * gNumFrameResources;

    D3D12_DESCRIPTOR_HEAP_DESC cbvHeapDesc;
    cbvHeapDesc.NumDescriptors = numDescriptors;
    cbvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    cbvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    cbvHeapDesc.NodeMask = 0;
    ThrowIfFailed(md3dDevice->CreateDescriptorHeap(&cbvHeapDesc,
        IID_PPV_ARGS(&mCbvHeap)));
}

//assuming we have n renter items, we can populate the CBV heap with the following code where descriptors 0 to n-
//1 contain the object CBVs for the 0th frame resource, descriptors n to 2n−1 contains the
//object CBVs for 1st frame resource, descriptors 2n to 3n−1 contain the objects CBVs for
//the 2nd frame resource, and descriptors 3n, 3n + 1, and 3n + 2 contain the pass CBVs for the
//0th, 1st, and 2nd frame resource
void ShapesApp::BuildConstantBufferViews()
{
    UINT objCBByteSize = d3dUtil::CalcConstantBufferByteSize(sizeof(ObjectConstants));

    UINT objCount = (UINT)mOpaqueRitems.size();

    // Need a CBV descriptor for each object for each frame resource.
    for(int frameIndex = 0; frameIndex < gNumFrameResources; ++frameIndex)
    {
        auto objectCB = mFrameResources[frameIndex]->ObjectCB->Resource();
        for(UINT i = 0; i < objCount; ++i)
        {
            D3D12_GPU_VIRTUAL_ADDRESS cbAddress = objectCB->GetGPUVirtualAddress();

            // Offset to the ith object constant buffer in the buffer.
            cbAddress += i*objCBByteSize;

            // Offset to the object cbv in the descriptor heap.
            int heapIndex = frameIndex*objCount + i;

			//we can get a handle to the first descriptor in a heap with the ID3D12DescriptorHeap::GetCPUDescriptorHandleForHeapStart
            auto handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(mCbvHeap->GetCPUDescriptorHandleForHeapStart());

			//our heap has more than one descriptor,we need to know the size to increment in the heap to get to the next descriptor
			//This is hardware specific, so we have to query this information from the device, and it depends on
			//the heap type.Recall that our D3DApp class caches this information: 	mCbvSrvUavDescriptorSize = md3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            handle.Offset(heapIndex, mCbvSrvUavDescriptorSize);

            D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
            cbvDesc.BufferLocation = cbAddress;
            cbvDesc.SizeInBytes = objCBByteSize;

            md3dDevice->CreateConstantBufferView(&cbvDesc, handle);
        }
    }

    UINT passCBByteSize = d3dUtil::CalcConstantBufferByteSize(sizeof(PassConstants));

    // Last three descriptors are the pass CBVs for each frame resource.
    for(int frameIndex = 0; frameIndex < gNumFrameResources; ++frameIndex)
    {
        auto passCB = mFrameResources[frameIndex]->PassCB->Resource();
        D3D12_GPU_VIRTUAL_ADDRESS cbAddress = passCB->GetGPUVirtualAddress();

        // Offset to the pass cbv in the descriptor heap.
        int heapIndex = mPassCbvOffset + frameIndex;
        auto handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(mCbvHeap->GetCPUDescriptorHandleForHeapStart());
        handle.Offset(heapIndex, mCbvSrvUavDescriptorSize);

        D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
        cbvDesc.BufferLocation = cbAddress;
        cbvDesc.SizeInBytes = passCBByteSize;
        
        md3dDevice->CreateConstantBufferView(&cbvDesc, handle);
    }
}

//A root signature defines what resources need to be bound to the pipeline before issuing a draw call and
//how those resources get mapped to shader input registers. there is a limit of 64 DWORDs that can be put in a root signature.
void ShapesApp::BuildRootSignature()
{
    CD3DX12_DESCRIPTOR_RANGE cbvTable0;
    cbvTable0.Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0);

    CD3DX12_DESCRIPTOR_RANGE cbvTable1;
    cbvTable1.Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 1);

	// Root parameter can be a table, root descriptor or root constants.
	CD3DX12_ROOT_PARAMETER slotRootParameter[2];

	// Create root CBVs.
    slotRootParameter[0].InitAsDescriptorTable(1, &cbvTable0);
    slotRootParameter[1].InitAsDescriptorTable(1, &cbvTable1);

	// A root signature is an array of root parameters.
	CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(2, slotRootParameter, 0, nullptr, 
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

	// create a root signature with a single slot which points to a descriptor range consisting of a single constant buffer
	ComPtr<ID3DBlob> serializedRootSig = nullptr;
	ComPtr<ID3DBlob> errorBlob = nullptr;
	HRESULT hr = D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1,
		serializedRootSig.GetAddressOf(), errorBlob.GetAddressOf());

	if(errorBlob != nullptr)
	{
		::OutputDebugStringA((char*)errorBlob->GetBufferPointer());
	}
	ThrowIfFailed(hr);

	ThrowIfFailed(md3dDevice->CreateRootSignature(
		0,
		serializedRootSig->GetBufferPointer(),
		serializedRootSig->GetBufferSize(),
		IID_PPV_ARGS(mRootSignature.GetAddressOf())));
}

void ShapesApp::BuildShadersAndInputLayout()
{
	mShaders["standardVS"] = d3dUtil::CompileShader(L"Shaders\\color.hlsl", nullptr, "VS", "vs_5_1");
	mShaders["opaquePS"] = d3dUtil::CompileShader(L"Shaders\\color.hlsl", nullptr, "PS", "ps_5_1");
	
    mInputLayout =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    };
}

void ShapesApp::BuildShapeGeometry()
{
    GeometryGenerator geoGen;
	GeometryGenerator::MeshData wall = geoGen.CreateBox(9.0f, 2.0f, 0.5f, 1);
	GeometryGenerator::MeshData grid = geoGen.CreateGrid(20.0f, 30.0f, 60, 40);
	GeometryGenerator::MeshData sphere = geoGen.CreateSphere(0.5f, 20, 20);
	GeometryGenerator::MeshData wallPillar = geoGen.CreateCylinder(1.0f, 1.0f, 3.0f, 4 , 4);
    GeometryGenerator::MeshData fountainPillar = geoGen.CreateCylinder(1.0f, 1.0f, 3.0f, 8, 8);
    GeometryGenerator::MeshData wallPillarTop = geoGen.CreateCylinder(1.0f, 0.0f, 3.0f, 4, 5);
    GeometryGenerator::MeshData fountainPillarTop = geoGen.CreateCylinder(1.0f, 0.0f, 1.0f, 8, 1);
    GeometryGenerator::MeshData centerFountain = geoGen.CreateCylinder(2.0f, 0.0f, 1.0f, 4, 5);
   
    // Create pyramid and createa octagonal prism dont work

	//
	// We are concatenating all the geometry into one big vertex/index buffer.  So
	// define the regions in the buffer each submesh covers.
	//

	// Cache the vertex offsets to each object in the concatenated vertex buffer.
	UINT wallVertexOffset = 0;
	UINT gridVertexOffset = (UINT)wall.Vertices.size();
	UINT sphereVertexOffset = gridVertexOffset + (UINT)grid.Vertices.size();
	UINT wallPillarVertexOffset = sphereVertexOffset + (UINT)sphere.Vertices.size();
    UINT fountainPillarVertexOffset = wallPillarVertexOffset + (UINT)wallPillar.Vertices.size();
    UINT wallPillarTopVertexOffset = fountainPillarVertexOffset + (UINT)fountainPillar.Vertices.size();
    UINT fountainPillarTopVertexOffset = wallPillarTopVertexOffset + (UINT)wallPillarTop.Vertices.size();
    UINT centerFountainVertexOffset = fountainPillarTopVertexOffset + (UINT)fountainPillarTop.Vertices.size();



	// Cache the starting index for each object in the concatenated index buffer.
    UINT wallIndexOffset = 0;
    UINT gridIndexOffset = (UINT)wall.Indices32.size();
    UINT sphereIndexOffset = gridIndexOffset + (UINT)grid.Indices32.size();
    UINT wallPillarIndexOffset = sphereIndexOffset + (UINT)sphere.Indices32.size();
    UINT fountainPillarIndexOffset = wallPillarIndexOffset + (UINT)wallPillar.Indices32.size();
    UINT wallPillarTopIndexOffset = fountainPillarIndexOffset + (UINT)fountainPillar.Indices32.size();
    UINT fountainPillarTopIndexOffset = wallPillarTopIndexOffset + (UINT)wallPillarTop.Indices32.size();
    UINT centerFountainIndexOffset = fountainPillarTopIndexOffset + (UINT)fountainPillarTop.Indices32.size();

    // Define the SubmeshGeometry that cover different 
    // regions of the vertex/index buffers.

	SubmeshGeometry wallSubmesh;
	wallSubmesh.IndexCount = (UINT)wall.Indices32.size();
	wallSubmesh.StartIndexLocation = wallIndexOffset;
    wallSubmesh.BaseVertexLocation = wallVertexOffset;

	SubmeshGeometry gridSubmesh;
	gridSubmesh.IndexCount = (UINT)grid.Indices32.size();
	gridSubmesh.StartIndexLocation = gridIndexOffset;
	gridSubmesh.BaseVertexLocation = gridVertexOffset;

	SubmeshGeometry sphereSubmesh;
	sphereSubmesh.IndexCount = (UINT)sphere.Indices32.size();
	sphereSubmesh.StartIndexLocation = sphereIndexOffset;
	sphereSubmesh.BaseVertexLocation = sphereVertexOffset;

	SubmeshGeometry wallPillarSubmesh;
	wallPillarSubmesh.IndexCount = (UINT)wallPillar.Indices32.size();
	wallPillarSubmesh.StartIndexLocation = wallPillarIndexOffset;
	wallPillarSubmesh.BaseVertexLocation = wallPillarVertexOffset;

    SubmeshGeometry fountainPillarSubmesh;
    fountainPillarSubmesh.IndexCount = (UINT)wall.Indices32.size();
    fountainPillarSubmesh.StartIndexLocation = fountainPillarIndexOffset;
    fountainPillarSubmesh.BaseVertexLocation = fountainPillarVertexOffset;

    SubmeshGeometry wallPillarTopSubmesh;
    wallPillarTopSubmesh.IndexCount = (UINT)wall.Indices32.size();
    wallPillarTopSubmesh.StartIndexLocation = wallPillarTopIndexOffset;
    wallPillarTopSubmesh.BaseVertexLocation = wallPillarTopVertexOffset;

    SubmeshGeometry fountainPillarTopSubmesh;
    fountainPillarTopSubmesh.IndexCount = (UINT)wall.Indices32.size();
    fountainPillarTopSubmesh.StartIndexLocation = fountainPillarTopIndexOffset;
    fountainPillarTopSubmesh.BaseVertexLocation = fountainPillarTopVertexOffset;

    SubmeshGeometry centerFountainSubmesh;
    centerFountainSubmesh.IndexCount = (UINT)wall.Indices32.size();
    centerFountainSubmesh.StartIndexLocation = centerFountainIndexOffset;
    centerFountainSubmesh.BaseVertexLocation = centerFountainVertexOffset;




	//
	// Extract the vertex elements we are interested in and pack the
	// vertices of all the meshes into one vertex buffer.
	//

	auto totalVertexCount =
		wall.Vertices.size() +
		grid.Vertices.size() +
		sphere.Vertices.size() +
		wallPillar.Vertices.size() + 
        fountainPillar.Vertices.size() +
        wallPillarTop.Vertices.size() +
        fountainPillarTop.Vertices.size() +
        centerFountain.Vertices.size();

	std::vector<Vertex> vertices(totalVertexCount);

    // COLORS HERE
	UINT k = 0;
	for(size_t i = 0; i < wall.Vertices.size(); ++i, ++k)
	{
		vertices[k].Pos = wall.Vertices[i].Position;
        vertices[k].Color = XMFLOAT4(DirectX::Colors::DarkGreen);
	}

	for(size_t i = 0; i < grid.Vertices.size(); ++i, ++k)
	{
		vertices[k].Pos = grid.Vertices[i].Position;
        vertices[k].Color = XMFLOAT4(DirectX::Colors::ForestGreen);
	}

	for(size_t i = 0; i < sphere.Vertices.size(); ++i, ++k)
	{
		vertices[k].Pos = sphere.Vertices[i].Position;
        vertices[k].Color = XMFLOAT4(DirectX::Colors::Crimson);
	}

	for(size_t i = 0; i < wallPillar.Vertices.size(); ++i, ++k)
	{
		vertices[k].Pos = wallPillar.Vertices[i].Position;
		vertices[k].Color = XMFLOAT4(DirectX::Colors::SteelBlue);
	}

    for (size_t i = 0; i < fountainPillar.Vertices.size(); ++i, ++k)
    {
        vertices[k].Pos = fountainPillar.Vertices[i].Position;
        vertices[k].Color = XMFLOAT4(DirectX::Colors::Aquamarine);
    }

    for (size_t i = 0; i < wallPillarTop.Vertices.size(); ++i, ++k)
    {
        vertices[k].Pos = wallPillarTop.Vertices[i].Position;
        vertices[k].Color = XMFLOAT4(DirectX::Colors::DarkSalmon);
    }

    for (size_t i = 0; i < fountainPillarTop.Vertices.size(); ++i, ++k)
    {
        vertices[k].Pos = fountainPillarTop.Vertices[i].Position;
        vertices[k].Color = XMFLOAT4(DirectX::Colors::Olive);
    }

    for (size_t i = 0; i < centerFountain.Vertices.size(); ++i, ++k)
    {
        vertices[k].Pos = centerFountain.Vertices[i].Position;
        vertices[k].Color = XMFLOAT4(DirectX::Colors::MintCream);
    }

	std::vector<std::uint16_t> indices;
	indices.insert(indices.end(), std::begin(wall.GetIndices16()), std::end(wall.GetIndices16()));
	indices.insert(indices.end(), std::begin(grid.GetIndices16()), std::end(grid.GetIndices16()));
	indices.insert(indices.end(), std::begin(sphere.GetIndices16()), std::end(sphere.GetIndices16()));
	indices.insert(indices.end(), std::begin(wallPillar.GetIndices16()), std::end(wallPillar.GetIndices16()));
    indices.insert(indices.end(), std::begin(fountainPillar.GetIndices16()), std::end(fountainPillar.GetIndices16()));
    indices.insert(indices.end(), std::begin(wallPillarTop.GetIndices16()), std::end(wallPillarTop.GetIndices16()));
    indices.insert(indices.end(), std::begin(fountainPillarTop.GetIndices16()), std::end(fountainPillarTop.GetIndices16()));
    indices.insert(indices.end(), std::begin(centerFountain.GetIndices16()), std::end(centerFountain.GetIndices16()));

    const UINT vbByteSize = (UINT)vertices.size() * sizeof(Vertex);
    const UINT ibByteSize = (UINT)indices.size()  * sizeof(std::uint16_t);

	auto geo = std::make_unique<MeshGeometry>();
	geo->Name = "shapeGeo";

	ThrowIfFailed(D3DCreateBlob(vbByteSize, &geo->VertexBufferCPU));
	CopyMemory(geo->VertexBufferCPU->GetBufferPointer(), vertices.data(), vbByteSize);

	ThrowIfFailed(D3DCreateBlob(ibByteSize, &geo->IndexBufferCPU));
	CopyMemory(geo->IndexBufferCPU->GetBufferPointer(), indices.data(), ibByteSize);

	geo->VertexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), vertices.data(), vbByteSize, geo->VertexBufferUploader);

	geo->IndexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), indices.data(), ibByteSize, geo->IndexBufferUploader);

	geo->VertexByteStride = sizeof(Vertex);
	geo->VertexBufferByteSize = vbByteSize;
	geo->IndexFormat = DXGI_FORMAT_R16_UINT;
	geo->IndexBufferByteSize = ibByteSize;

	geo->DrawArgs["wall"] = wallSubmesh;
	geo->DrawArgs["grid"] = gridSubmesh;
	geo->DrawArgs["sphere"] = sphereSubmesh;
	geo->DrawArgs["wallPillar"] = wallPillarSubmesh;
    geo->DrawArgs["fountainPillar"] = fountainPillarSubmesh;
    geo->DrawArgs["wallPillarTop"] = wallPillarTopSubmesh;
    geo->DrawArgs["fountainPillarTop"] = fountainPillarTopSubmesh;
    geo->DrawArgs["centerFountain"] = centerFountainSubmesh;

	mGeometries[geo->Name] = std::move(geo);
}

void ShapesApp::BuildPSOs()
{
    D3D12_GRAPHICS_PIPELINE_STATE_DESC opaquePsoDesc;

	//
	// PSO for opaque objects.
	//
    ZeroMemory(&opaquePsoDesc, sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));
	opaquePsoDesc.InputLayout = { mInputLayout.data(), (UINT)mInputLayout.size() };
	opaquePsoDesc.pRootSignature = mRootSignature.Get();
	opaquePsoDesc.VS = 
	{ 
		reinterpret_cast<BYTE*>(mShaders["standardVS"]->GetBufferPointer()), 
		mShaders["standardVS"]->GetBufferSize()
	};
	opaquePsoDesc.PS = 
	{ 
		reinterpret_cast<BYTE*>(mShaders["opaquePS"]->GetBufferPointer()),
		mShaders["opaquePS"]->GetBufferSize()
	};
	opaquePsoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    opaquePsoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
	opaquePsoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	opaquePsoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
	opaquePsoDesc.SampleMask = UINT_MAX;
	opaquePsoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	opaquePsoDesc.NumRenderTargets = 1;
	opaquePsoDesc.RTVFormats[0] = mBackBufferFormat;
	opaquePsoDesc.SampleDesc.Count = m4xMsaaState ? 4 : 1;
	opaquePsoDesc.SampleDesc.Quality = m4xMsaaState ? (m4xMsaaQuality - 1) : 0;
	opaquePsoDesc.DSVFormat = mDepthStencilFormat;
    ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&opaquePsoDesc, IID_PPV_ARGS(&mPSOs["opaque"])));


    //
    // PSO for opaque wireframe objects.
    //

    D3D12_GRAPHICS_PIPELINE_STATE_DESC opaqueWireframePsoDesc = opaquePsoDesc;
    opaqueWireframePsoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_WIREFRAME;
    ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&opaqueWireframePsoDesc, IID_PPV_ARGS(&mPSOs["opaque_wireframe"])));
}

void ShapesApp::BuildFrameResources()
{
    for(int i = 0; i < gNumFrameResources; ++i)
    {
        mFrameResources.push_back(std::make_unique<FrameResource>(md3dDevice.Get(),
            1, (UINT)mAllRitems.size()));
    }
}

void ShapesApp::BuildRenderItems()
{

    XMVECTOR yAxis = { 0.0f,1.0f,0.0f };
    UINT objCBIndex = 3;
    float degreeRotation45 = 0.785398;
    float degreeRotation90 = 1.5708;

    // grid
    auto gridRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&gridRitem->World, XMMatrixScaling(2.0f, 2.0f, 2.0f) * XMMatrixTranslation(0.0f, -1.0f, 0.0f));
    gridRitem->ObjCBIndex = 0;
    gridRitem->Geo = mGeometries["shapeGeo"].get();
    gridRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    gridRitem->IndexCount = gridRitem->Geo->DrawArgs["grid"].IndexCount;
    gridRitem->StartIndexLocation = gridRitem->Geo->DrawArgs["grid"].StartIndexLocation;
    gridRitem->BaseVertexLocation = gridRitem->Geo->DrawArgs["grid"].BaseVertexLocation;
    mAllRitems.push_back(std::move(gridRitem));

    //center fountain
    auto centerFountainRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&centerFountainRitem->World, XMMatrixScaling(2.0f, 2.0f, 2.0f) * XMMatrixTranslation(0.0f, 0.0f, 0.0f));
    centerFountainRitem->ObjCBIndex = 1;
    centerFountainRitem->Geo = mGeometries["shapeGeo"].get();
    centerFountainRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    centerFountainRitem->IndexCount = centerFountainRitem->Geo->DrawArgs["centerFountain"].IndexCount;
    centerFountainRitem->StartIndexLocation = centerFountainRitem->Geo->DrawArgs["centerFountain"].StartIndexLocation;
    centerFountainRitem->BaseVertexLocation = centerFountainRitem->Geo->DrawArgs["centerFountain"].BaseVertexLocation;
    mAllRitems.push_back(std::move(centerFountainRitem));

    // fountain piece
    auto fountainSphereRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&fountainSphereRitem->World, XMMatrixScaling(2.0f, 2.0f, 2.0f) * XMMatrixTranslation(0.0f, 2.0f, 0.0f));
    fountainSphereRitem->ObjCBIndex = 2;
    fountainSphereRitem->Geo = mGeometries["shapeGeo"].get();
    fountainSphereRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    fountainSphereRitem->IndexCount = fountainSphereRitem->Geo->DrawArgs["sphere"].IndexCount;
    fountainSphereRitem->StartIndexLocation = fountainSphereRitem->Geo->DrawArgs["sphere"].StartIndexLocation;
    fountainSphereRitem->BaseVertexLocation = fountainSphereRitem->Geo->DrawArgs["sphere"].BaseVertexLocation;
    mAllRitems.push_back(std::move(fountainSphereRitem));

    // 4 walls
    for (int i = 0; i < 1; ++i)
    {
        auto wallRitemFront = std::make_unique<RenderItem>();
        auto wallRitemBack = std::make_unique<RenderItem>();
        auto wallRitemLeft = std::make_unique<RenderItem>();
        auto wallRitemRight = std::make_unique<RenderItem>();

        //XMStoreFloat4x4(&wallRitemFront->World, XMMatrixScaling(2.0f, 2.0f, 2.0f) * XMMatrixTranslation(0.0f, 1.0f, -10.0f) * XMMatrixRotationAxis(yAxis, 4.7));

        XMStoreFloat4x4(&wallRitemFront->World, XMMatrixScaling(2.0f, 2.0f, 2.0f) * XMMatrixTranslation(0.0f, 1.0f, -10.0f));
        wallRitemFront->ObjCBIndex = objCBIndex++;
        wallRitemFront->Geo = mGeometries["shapeGeo"].get();
        wallRitemFront->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        wallRitemFront->IndexCount = wallRitemFront->Geo->DrawArgs["wall"].IndexCount;
        wallRitemFront->StartIndexLocation = wallRitemFront->Geo->DrawArgs["wall"].StartIndexLocation;
        wallRitemFront->BaseVertexLocation = wallRitemFront->Geo->DrawArgs["wall"].BaseVertexLocation;
        mAllRitems.push_back(std::move(wallRitemFront));

        XMStoreFloat4x4(&wallRitemBack->World, XMMatrixScaling(2.0f, 2.0f, 2.0f) * XMMatrixTranslation(0.0f, 1.0f, 10.0f));
        wallRitemBack->ObjCBIndex = objCBIndex++;
        wallRitemBack->Geo = mGeometries["shapeGeo"].get();
        wallRitemBack->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        wallRitemBack->IndexCount = wallRitemBack->Geo->DrawArgs["wall"].IndexCount;
        wallRitemBack->StartIndexLocation = wallRitemBack->Geo->DrawArgs["wall"].StartIndexLocation;
        wallRitemBack->BaseVertexLocation = wallRitemBack->Geo->DrawArgs["wall"].BaseVertexLocation;
        mAllRitems.push_back(std::move(wallRitemBack));

        XMStoreFloat4x4(&wallRitemLeft->World, XMMatrixScaling(2.0f, 2.0f, 2.0f) * XMMatrixTranslation(0.0f, 1.0f, 10.0f) * XMMatrixRotationAxis(yAxis, degreeRotation90));
        wallRitemLeft->ObjCBIndex = objCBIndex++;
        wallRitemLeft->Geo = mGeometries["shapeGeo"].get();
        wallRitemLeft->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        wallRitemLeft->IndexCount = wallRitemLeft->Geo->DrawArgs["wall"].IndexCount;
        wallRitemLeft->StartIndexLocation = wallRitemLeft->Geo->DrawArgs["wall"].StartIndexLocation;
        wallRitemLeft->BaseVertexLocation = wallRitemLeft->Geo->DrawArgs["wall"].BaseVertexLocation;
        mAllRitems.push_back(std::move(wallRitemLeft));

        XMStoreFloat4x4(&wallRitemRight->World, XMMatrixScaling(2.0f, 2.0f, 2.0f) * XMMatrixTranslation(0.0f, 1.0f, -10.0f) * XMMatrixRotationAxis(yAxis, degreeRotation90));
        wallRitemRight->ObjCBIndex = objCBIndex++;
        wallRitemRight->Geo = mGeometries["shapeGeo"].get();
        wallRitemRight->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        wallRitemRight->IndexCount = wallRitemRight->Geo->DrawArgs["wall"].IndexCount;
        wallRitemRight->StartIndexLocation = wallRitemRight->Geo->DrawArgs["wall"].StartIndexLocation;
        wallRitemRight->BaseVertexLocation = wallRitemRight->Geo->DrawArgs["wall"].BaseVertexLocation;
        mAllRitems.push_back(std::move(wallRitemRight));




        


    }
    // 4 pillars
    for (int i = 0; i < 1; ++i)
    {
        auto wallPillarFLRitem = std::make_unique<RenderItem>();
        auto wallPillarFRRitem = std::make_unique<RenderItem>();
        auto wallPillarBLRitem = std::make_unique<RenderItem>();
        auto wallPillarBRRitem = std::make_unique<RenderItem>();

        //XMStoreFloat4x4(&wallRitemFront->World, XMMatrixScaling(2.0f, 2.0f, 2.0f) * XMMatrixTranslation(0.0f, 1.0f, -10.0f) * XMMatrixRotationAxis(yAxis, 1.57));

        XMStoreFloat4x4(&wallPillarFLRitem->World, XMMatrixScaling(2.0f, 2.0f, 2.0f) * XMMatrixTranslation(0.0f, 2.0f, -13.0f) * XMMatrixRotationAxis(yAxis, degreeRotation45));
        wallPillarFLRitem->ObjCBIndex = objCBIndex++;
        wallPillarFLRitem->Geo = mGeometries["shapeGeo"].get();
        wallPillarFLRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        wallPillarFLRitem->IndexCount = wallPillarFLRitem->Geo->DrawArgs["wallPillar"].IndexCount;
        wallPillarFLRitem->StartIndexLocation = wallPillarFLRitem->Geo->DrawArgs["wallPillar"].StartIndexLocation;
        wallPillarFLRitem->BaseVertexLocation = wallPillarFLRitem->Geo->DrawArgs["wallPillar"].BaseVertexLocation;
        mAllRitems.push_back(std::move(wallPillarFLRitem));

        XMStoreFloat4x4(&wallPillarFRRitem->World, XMMatrixScaling(2.0f, 2.0f, 2.0f) * XMMatrixTranslation(13.0f, 2.0f, 0.0f) * XMMatrixRotationAxis(yAxis, degreeRotation45));
        wallPillarFRRitem->ObjCBIndex = objCBIndex++;
        wallPillarFRRitem->Geo = mGeometries["shapeGeo"].get();
        wallPillarFRRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        wallPillarFRRitem->IndexCount = wallPillarFRRitem->Geo->DrawArgs["wallPillar"].IndexCount;
        wallPillarFRRitem->StartIndexLocation = wallPillarFRRitem->Geo->DrawArgs["wallPillar"].StartIndexLocation;
        wallPillarFRRitem->BaseVertexLocation = wallPillarFRRitem->Geo->DrawArgs["wallPillar"].BaseVertexLocation;
        mAllRitems.push_back(std::move(wallPillarFRRitem));

        XMStoreFloat4x4(&wallPillarBLRitem->World, XMMatrixScaling(2.0f, 2.0f, 2.0f) * XMMatrixTranslation(-13.0f, 2.0f, 0.0f) * XMMatrixRotationAxis(yAxis, degreeRotation45));
        wallPillarBLRitem->ObjCBIndex = objCBIndex++;
        wallPillarBLRitem->Geo = mGeometries["shapeGeo"].get();
        wallPillarBLRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        wallPillarBLRitem->IndexCount = wallPillarBLRitem->Geo->DrawArgs["wallPillar"].IndexCount;
        wallPillarBLRitem->StartIndexLocation = wallPillarBLRitem->Geo->DrawArgs["wallPillar"].StartIndexLocation;
        wallPillarBLRitem->BaseVertexLocation = wallPillarBLRitem->Geo->DrawArgs["wallPillar"].BaseVertexLocation;
        mAllRitems.push_back(std::move(wallPillarBLRitem));

        XMStoreFloat4x4(&wallPillarBRRitem->World, XMMatrixScaling(2.0f, 2.0f, 2.0f) * XMMatrixTranslation(0.0f, 2.0f, 13.0f) * XMMatrixRotationAxis(yAxis, degreeRotation45));
        wallPillarBRRitem->ObjCBIndex = objCBIndex++;
        wallPillarBRRitem->Geo = mGeometries["shapeGeo"].get();
        wallPillarBRRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        wallPillarBRRitem->IndexCount = wallPillarBRRitem->Geo->DrawArgs["wallPillar"].IndexCount;
        wallPillarBRRitem->StartIndexLocation = wallPillarBRRitem->Geo->DrawArgs["wallPillar"].StartIndexLocation;
        wallPillarBRRitem->BaseVertexLocation = wallPillarBRRitem->Geo->DrawArgs["wallPillar"].BaseVertexLocation;
        mAllRitems.push_back(std::move(wallPillarBRRitem));

      
    }
    // 4 pillar tops
    for (int i = 0; i < 1; ++i)
    {
        auto wallPillarFLTopRitem = std::make_unique<RenderItem>();
        auto wallPillarFRTopRitem = std::make_unique<RenderItem>();
        auto wallPillarBLTopRitem = std::make_unique<RenderItem>();
        auto wallPillarBRTopRitem = std::make_unique<RenderItem>();

        //XMStoreFloat4x4(&wallRitemFront->World, XMMatrixScaling(2.0f, 2.0f, 2.0f) * XMMatrixTranslation(0.0f, 1.0f, -10.0f) * XMMatrixRotationAxis(yAxis, 1.57));

        XMStoreFloat4x4(&wallPillarFLTopRitem->World, XMMatrixScaling(2.0f, 2.0f, 2.0f)* XMMatrixTranslation(0.0f, 7.0f, -13.0f)* XMMatrixRotationAxis(yAxis, degreeRotation45));
        wallPillarFLTopRitem->ObjCBIndex = objCBIndex++;
        wallPillarFLTopRitem->Geo = mGeometries["shapeGeo"].get();
        wallPillarFLTopRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        wallPillarFLTopRitem->IndexCount = wallPillarFLTopRitem->Geo->DrawArgs["wallPillarTop"].IndexCount;
        wallPillarFLTopRitem->StartIndexLocation = wallPillarFLTopRitem->Geo->DrawArgs["wallPillarTop"].StartIndexLocation;
        wallPillarFLTopRitem->BaseVertexLocation = wallPillarFLTopRitem->Geo->DrawArgs["wallPillarTop"].BaseVertexLocation;
        mAllRitems.push_back(std::move(wallPillarFLTopRitem));

        XMStoreFloat4x4(&wallPillarFRTopRitem->World, XMMatrixScaling(2.0f, 2.0f, 2.0f)* XMMatrixTranslation(13.0f, 7.0f, 0.0f)* XMMatrixRotationAxis(yAxis, degreeRotation45));
        wallPillarFRTopRitem->ObjCBIndex = objCBIndex++;
        wallPillarFRTopRitem->Geo = mGeometries["shapeGeo"].get();
        wallPillarFRTopRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        wallPillarFRTopRitem->IndexCount = wallPillarFRTopRitem->Geo->DrawArgs["wallPillarTop"].IndexCount;
        wallPillarFRTopRitem->StartIndexLocation = wallPillarFRTopRitem->Geo->DrawArgs["wallPillarTop"].StartIndexLocation;
        wallPillarFRTopRitem->BaseVertexLocation = wallPillarFRTopRitem->Geo->DrawArgs["wallPillarTop"].BaseVertexLocation;
        mAllRitems.push_back(std::move(wallPillarFRTopRitem));

        XMStoreFloat4x4(&wallPillarBLTopRitem->World, XMMatrixScaling(2.0f, 2.0f, 2.0f)* XMMatrixTranslation(-13.0f, 7.0f, 0.0f)* XMMatrixRotationAxis(yAxis, degreeRotation45));
        wallPillarBLTopRitem->ObjCBIndex = objCBIndex++;
        wallPillarBLTopRitem->Geo = mGeometries["shapeGeo"].get();
        wallPillarBLTopRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        wallPillarBLTopRitem->IndexCount = wallPillarBLTopRitem->Geo->DrawArgs["wallPillarTop"].IndexCount;
        wallPillarBLTopRitem->StartIndexLocation = wallPillarBLTopRitem->Geo->DrawArgs["wallPillarTop"].StartIndexLocation;
        wallPillarBLTopRitem->BaseVertexLocation = wallPillarBLTopRitem->Geo->DrawArgs["wallPillarTop"].BaseVertexLocation;
        mAllRitems.push_back(std::move(wallPillarBLTopRitem));

        XMStoreFloat4x4(&wallPillarBRTopRitem->World, XMMatrixScaling(2.0f, 2.0f, 2.0f)* XMMatrixTranslation(0.0f, 7.0f, 13.0f)* XMMatrixRotationAxis(yAxis, degreeRotation45));
        wallPillarBRTopRitem->ObjCBIndex = objCBIndex++;
        wallPillarBRTopRitem->Geo = mGeometries["shapeGeo"].get();
        wallPillarBRTopRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        wallPillarBRTopRitem->IndexCount = wallPillarBRTopRitem->Geo->DrawArgs["wallPillarTop"].IndexCount;
        wallPillarBRTopRitem->StartIndexLocation = wallPillarBRTopRitem->Geo->DrawArgs["wallPillarTop"].StartIndexLocation;
        wallPillarBRTopRitem->BaseVertexLocation = wallPillarBRTopRitem->Geo->DrawArgs["wallPillarTop"].BaseVertexLocation;
        mAllRitems.push_back(std::move(wallPillarBRTopRitem));
    }
    // Center pillars
    for (int i = 0; i < 1; ++i)
    {
        auto centerPillarFrontRitem = std::make_unique<RenderItem>();
        auto centerPillarBackRitem = std::make_unique<RenderItem>();
        auto centerPillarLeftRitem = std::make_unique<RenderItem>();
        auto centerPillarRightRitem = std::make_unique<RenderItem>();

        //XMStoreFloat4x4(&wallRitemFront->World, XMMatrixScaling(2.0f, 2.0f, 2.0f) * XMMatrixTranslation(0.0f, 1.0f, -10.0f) * XMMatrixRotationAxis(yAxis, 1.57));

        XMStoreFloat4x4(&centerPillarFrontRitem->World, XMMatrixScaling(0.5f, 3.0f, 0.5f)* XMMatrixTranslation(3.0f, 3.5f, -3.0f)* XMMatrixRotationAxis(yAxis, degreeRotation45));
        centerPillarFrontRitem->ObjCBIndex = objCBIndex++;
        centerPillarFrontRitem->Geo = mGeometries["shapeGeo"].get();
        centerPillarFrontRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        centerPillarFrontRitem->IndexCount = centerPillarFrontRitem->Geo->DrawArgs["fountainPillar"].IndexCount;
        centerPillarFrontRitem->StartIndexLocation = centerPillarFrontRitem->Geo->DrawArgs["fountainPillar"].StartIndexLocation;
        centerPillarFrontRitem->BaseVertexLocation = centerPillarFrontRitem->Geo->DrawArgs["fountainPillar"].BaseVertexLocation;
        mAllRitems.push_back(std::move(centerPillarFrontRitem));

        XMStoreFloat4x4(&centerPillarBackRitem->World, XMMatrixScaling(0.5f, 3.0f, 0.5f)* XMMatrixTranslation(-3.0f, 3.5f, 3.0f)* XMMatrixRotationAxis(yAxis, degreeRotation45));
        centerPillarBackRitem->ObjCBIndex = objCBIndex++;
        centerPillarBackRitem->Geo = mGeometries["shapeGeo"].get();
        centerPillarBackRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        centerPillarBackRitem->IndexCount = centerPillarBackRitem->Geo->DrawArgs["fountainPillar"].IndexCount;
        centerPillarBackRitem->StartIndexLocation = centerPillarBackRitem->Geo->DrawArgs["fountainPillar"].StartIndexLocation;
        centerPillarBackRitem->BaseVertexLocation = centerPillarBackRitem->Geo->DrawArgs["fountainPillar"].BaseVertexLocation;
        mAllRitems.push_back(std::move(centerPillarBackRitem));
       
        XMStoreFloat4x4(&centerPillarLeftRitem->World, XMMatrixScaling(0.5f, 3.0f, 0.5f) * XMMatrixTranslation(-3.0f, 3.5f, -3.0f) * XMMatrixRotationAxis(yAxis, degreeRotation45));
        centerPillarLeftRitem->ObjCBIndex = objCBIndex++;
        centerPillarLeftRitem->Geo = mGeometries["shapeGeo"].get();
        centerPillarLeftRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        centerPillarLeftRitem->IndexCount = centerPillarLeftRitem->Geo->DrawArgs["fountainPillar"].IndexCount;
        centerPillarLeftRitem->StartIndexLocation = centerPillarLeftRitem->Geo->DrawArgs["fountainPillar"].StartIndexLocation;
        centerPillarLeftRitem->BaseVertexLocation = centerPillarLeftRitem->Geo->DrawArgs["fountainPillar"].BaseVertexLocation;
        mAllRitems.push_back(std::move(centerPillarLeftRitem));

        XMStoreFloat4x4(&centerPillarRightRitem->World, XMMatrixScaling(0.5f, 3.0f, 0.5f)* XMMatrixTranslation(3.0f, 3.5f, 3.0f)* XMMatrixRotationAxis(yAxis, degreeRotation45));
        centerPillarRightRitem->ObjCBIndex = objCBIndex++;
        centerPillarRightRitem->Geo = mGeometries["shapeGeo"].get();
        centerPillarRightRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        centerPillarRightRitem->IndexCount = centerPillarRightRitem->Geo->DrawArgs["fountainPillar"].IndexCount;
        centerPillarRightRitem->StartIndexLocation = centerPillarRightRitem->Geo->DrawArgs["fountainPillar"].StartIndexLocation;
        centerPillarRightRitem->BaseVertexLocation = centerPillarRightRitem->Geo->DrawArgs["fountainPillar"].BaseVertexLocation;
        mAllRitems.push_back(std::move(centerPillarRightRitem));

    }
    // center pillar tops
    for (int i = 0; i < 1; ++i)
    {
        auto centerPillarFrontTopRitem = std::make_unique<RenderItem>();
        auto centerPillarBackTopRitem = std::make_unique<RenderItem>();
        auto centerPillarLeftTopRitem = std::make_unique<RenderItem>();
        auto centerPillarRightTopRitem = std::make_unique<RenderItem>();

        //XMStoreFloat4x4(&wallRitemFront->World, XMMatrixScaling(2.0f, 2.0f, 2.0f) * XMMatrixTranslation(0.0f, 1.0f, -10.0f) * XMMatrixRotationAxis(yAxis, 1.57));

        XMStoreFloat4x4(&centerPillarFrontTopRitem->World, XMMatrixScaling(0.5f, 0.5f, 0.5f) * XMMatrixTranslation(3.0f, 2.6f, -3.0f) * XMMatrixRotationAxis(yAxis, degreeRotation45));
        centerPillarFrontTopRitem->ObjCBIndex = objCBIndex++;
        centerPillarFrontTopRitem->Geo = mGeometries["shapeGeo"].get();
        centerPillarFrontTopRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        centerPillarFrontTopRitem->IndexCount = centerPillarFrontTopRitem->Geo->DrawArgs["fountainPillarTop"].IndexCount;
        centerPillarFrontTopRitem->StartIndexLocation = centerPillarFrontTopRitem->Geo->DrawArgs["fountainPillarTop"].StartIndexLocation;
        centerPillarFrontTopRitem->BaseVertexLocation = centerPillarFrontTopRitem->Geo->DrawArgs["fountainPillarTop"].BaseVertexLocation;
        mAllRitems.push_back(std::move(centerPillarFrontTopRitem));

        XMStoreFloat4x4(&centerPillarBackTopRitem->World, XMMatrixScaling(0.5f, 0.5f, 0.5f)* XMMatrixTranslation(-3.0f, 2.6f, 3.0f)* XMMatrixRotationAxis(yAxis, degreeRotation45));
        centerPillarBackTopRitem->ObjCBIndex = objCBIndex++;
        centerPillarBackTopRitem->Geo = mGeometries["shapeGeo"].get();
        centerPillarBackTopRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        centerPillarBackTopRitem->IndexCount = centerPillarBackTopRitem->Geo->DrawArgs["fountainPillarTop"].IndexCount;
        centerPillarBackTopRitem->StartIndexLocation = centerPillarBackTopRitem->Geo->DrawArgs["fountainPillarTop"].StartIndexLocation;
        centerPillarBackTopRitem->BaseVertexLocation = centerPillarBackTopRitem->Geo->DrawArgs["fountainPillarTop"].BaseVertexLocation;
        mAllRitems.push_back(std::move(centerPillarBackTopRitem));

        XMStoreFloat4x4(&centerPillarLeftTopRitem->World, XMMatrixScaling(0.5f, 0.5f, 0.5f)* XMMatrixTranslation(-3.0f, 2.6f, -3.0f)* XMMatrixRotationAxis(yAxis, degreeRotation45));
        centerPillarLeftTopRitem->ObjCBIndex = objCBIndex++;
        centerPillarLeftTopRitem->Geo = mGeometries["shapeGeo"].get();
        centerPillarLeftTopRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        centerPillarLeftTopRitem->IndexCount = centerPillarLeftTopRitem->Geo->DrawArgs["fountainPillarTop"].IndexCount;
        centerPillarLeftTopRitem->StartIndexLocation = centerPillarLeftTopRitem->Geo->DrawArgs["fountainPillarTop"].StartIndexLocation;
        centerPillarLeftTopRitem->BaseVertexLocation = centerPillarLeftTopRitem->Geo->DrawArgs["fountainPillarTop"].BaseVertexLocation;
        mAllRitems.push_back(std::move(centerPillarLeftTopRitem));

        XMStoreFloat4x4(&centerPillarRightTopRitem->World, XMMatrixScaling(0.5f, 0.5f, 0.5f)* XMMatrixTranslation(3.0f, 2.6f, 3.0f)* XMMatrixRotationAxis(yAxis, degreeRotation45));
        centerPillarRightTopRitem->ObjCBIndex = objCBIndex++;
        centerPillarRightTopRitem->Geo = mGeometries["shapeGeo"].get();
        centerPillarRightTopRitem->PrimitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        centerPillarRightTopRitem->IndexCount = centerPillarRightTopRitem->Geo->DrawArgs["fountainPillarTop"].IndexCount;
        centerPillarRightTopRitem->StartIndexLocation = centerPillarRightTopRitem->Geo->DrawArgs["fountainPillarTop"].StartIndexLocation;
        centerPillarRightTopRitem->BaseVertexLocation = centerPillarRightTopRitem->Geo->DrawArgs["fountainPillarTop"].BaseVertexLocation;
        mAllRitems.push_back(std::move(centerPillarRightTopRitem));

      
    }
	// All the render items are opaque.
	for(auto& e : mAllRitems)
		mOpaqueRitems.push_back(e.get());
}


//The DrawRenderItems method is invoked in the main Draw call:
void ShapesApp::DrawRenderItems(ID3D12GraphicsCommandList* cmdList, const std::vector<RenderItem*>& ritems)
{
    UINT objCBByteSize = d3dUtil::CalcConstantBufferByteSize(sizeof(ObjectConstants));
 
	auto objectCB = mCurrFrameResource->ObjectCB->Resource();

    // For each render item...
    for(size_t i = 0; i < ritems.size(); ++i)
    {
        auto ri = ritems[i];

        cmdList->IASetVertexBuffers(0, 1, &ri->Geo->VertexBufferView());
        cmdList->IASetIndexBuffer(&ri->Geo->IndexBufferView());
        cmdList->IASetPrimitiveTopology(ri->PrimitiveType);

        // Offset to the CBV in the descriptor heap for this object and for this frame resource.
        UINT cbvIndex = mCurrFrameResourceIndex*(UINT)mOpaqueRitems.size() + ri->ObjCBIndex;
        auto cbvHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(mCbvHeap->GetGPUDescriptorHandleForHeapStart());
        cbvHandle.Offset(cbvIndex, mCbvSrvUavDescriptorSize);

        cmdList->SetGraphicsRootDescriptorTable(0, cbvHandle);

        cmdList->DrawIndexedInstanced(ri->IndexCount, 1, ri->StartIndexLocation, ri->BaseVertexLocation, 0);
    }
}


