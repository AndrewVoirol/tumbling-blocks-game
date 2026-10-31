import { createBeveledBlockMesh, createSimpleBoxMesh } from '../geometry/beveledBlockMesh';
import {
  mat4Create,
  mat4FromRigidInverse,
  mat4FromRotationTranslationScale,
  mat4LookAt,
  mat4Multiply,
  mat4OrthoZO,
  Vec3,
} from '../math/vecMat';
import {
  LIGHTING_PRESETS,
  LightingPreset,
  STAINED_GLASS_PALETTE,
} from '../physics/constants';
import { PhysicsWorld } from '../physics/PhysicsWorld';
import { CAUSTIC_PASS_WGSL } from './shaders/causticPass.wgsl';
import { ENVIRONMENT_PASS_WGSL } from './shaders/environmentPass.wgsl';
import { GLASS_SHARD_WGSL } from './shaders/glassShard.wgsl';
import { GLASS_TOWER_WGSL } from './shaders/glassTower.wgsl';
import { OVERLAY_PASS_WGSL } from './shaders/overlayPass.wgsl';
import { POST_PROCESS_WGSL } from './shaders/postProcess.wgsl';
import { OrbitCamera } from '../engine/OrbitCamera';

const CAUSTIC_MAP_SIZE = 2048;
const MAX_BLOCKS_GPU = 64;
const BLOCK_FLOATS = 48; // 192 bytes per GPUBlockData
const MAX_SHARDS_GPU = 256;
const SHARD_FLOATS = 24; // 96 bytes per GPUShardInstance
const MAX_SHARD_VERTICES = 65536;
const SHARD_VERTEX_FLOATS = 10; // 40 bytes per shard vertex
const MAX_SPLINTERS_GPU = 400; // 320 splinters + 64 floating cathedral dust motes
const SPLINTER_FLOATS = 8; // 32 bytes per GPUSplinter

export class WebGPURenderer {
  public device!: GPUDevice;
  private context!: GPUCanvasContext;
  private canvasFormat!: GPUTextureFormat;
  private canvas: HTMLCanvasElement;

  // Static & Pre-allocated GPU Buffers
  private sceneUniformBuffer!: GPUBuffer;
  private blockStorageBuffer!: GPUBuffer;
  private shardStorageBuffer!: GPUBuffer;
  private splinterStorageBuffer!: GPUBuffer;
  private overlaySlotBuffer!: GPUBuffer;
  private springUniformBuffer!: GPUBuffer;

  private postUniformExtract!: GPUBuffer;
  private postUniformBlurH!: GPUBuffer;
  private postUniformBlurV!: GPUBuffer;
  private postUniformComposite!: GPUBuffer;

  private beveledVertexBuffer!: GPUBuffer;
  private beveledIndexBuffer!: GPUBuffer;
  private beveledIndexCount: number = 0;

  private boxVertexBuffer!: GPUBuffer;
  private boxIndexBuffer!: GPUBuffer;
  private boxIndexCount: number = 0;

  private shardVertexBuffer!: GPUBuffer;

  // Render Targets
  private causticTex!: GPUTexture;
  private causticView!: GPUTextureView;

  private hdrColorMSAA!: GPUTexture;
  private hdrColorMSAAView!: GPUTextureView;
  private hdrResolveTex!: GPUTexture;
  private hdrResolveView!: GPUTextureView;
  private refractionSnapshotTex!: GPUTexture;
  private refractionSnapshotView!: GPUTextureView;
  private depthMSAA!: GPUTexture;
  private depthMSAAView!: GPUTextureView;

  private bloomTexA!: GPUTexture;
  private bloomViewA!: GPUTextureView;
  private bloomTexB!: GPUTexture;
  private bloomViewB!: GPUTextureView;

  private linearSampler!: GPUSampler;

  // Render Pipelines
  private pipeBlockCaustic!: GPURenderPipeline;
  private pipeShardCaustic!: GPURenderPipeline;
  private pipeSky!: GPURenderPipeline;
  private pipeTable!: GPURenderPipeline;
  private pipeGlassTower!: GPURenderPipeline;
  private pipeGlassShard!: GPURenderPipeline;
  private pipeSplinters!: GPURenderPipeline;
  private pipeSlotGuide!: GPURenderPipeline;
  private pipeSpringLine!: GPURenderPipeline;
  private pipeBloomExtract!: GPURenderPipeline;
  private pipeBloomBlur!: GPURenderPipeline;
  private pipeComposite!: GPURenderPipeline;

  // Pre-allocated Bind Groups (zero allocation inside rAF!)
  private bgBlockCaustic!: GPUBindGroup;
  private bgShardCaustic!: GPUBindGroup;
  private bgSky!: GPUBindGroup;
  private bgTable!: GPUBindGroup;
  private bgGlassTower!: GPUBindGroup;
  private bgGlassShard!: GPUBindGroup;
  private bgSplinters!: GPUBindGroup;
  private bgSlotGuide!: GPUBindGroup;
  private bgSpringLine!: GPUBindGroup;
  private bgBloomExtract!: GPUBindGroup;
  private bgBloomBlurH!: GPUBindGroup;
  private bgBloomBlurV!: GPUBindGroup;
  private bgComposite!: GPUBindGroup;

  // Pre-allocated CPU Float32Array staging views
  private sceneUniformData = new Float32Array(64);
  private blockStorageData = new Float32Array(MAX_BLOCKS_GPU * BLOCK_FLOATS);
  private shardStorageData = new Float32Array(MAX_SHARDS_GPU * SHARD_FLOATS);
  private splinterStorageData = new Float32Array(MAX_SPLINTERS_GPU * SPLINTER_FLOATS);
  private overlaySlotData = new Float32Array(8 * 20); // up to 8 slots * 80 bytes
  private springUniformData = new Float32Array(8);

  private sunViewMatrix = mat4Create();
  private sunProjMatrix = mat4Create();
  private sunViewProjMatrix = mat4Create();
  private tmpMatA = mat4Create();
  private tmpMatB = mat4Create();

  public lightingPreset: LightingPreset = LIGHTING_PRESETS[0];
  private width: number = 0;
  private height: number = 0;
  private destroyed: boolean = false;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
  }

  public async init(): Promise<void> {
    if (!navigator.gpu) {
      throw new Error('WebGPU is not supported in this browser.');
    }

    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });
    if (!adapter) {
      throw new Error('Unable to acquire a WebGPU graphics adapter.');
    }

    this.device = await adapter.requestDevice();
    const ctx = this.canvas.getContext('webgpu');
    if (!ctx) {
      throw new Error('Unable to configure WebGPU canvas context.');
    }
    this.context = ctx;
    this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();

    this.context.configure({
      device: this.device,
      format: this.canvasFormat,
      alphaMode: 'opaque',
    });

    this.linearSampler = this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    this.createStaticBuffersAndMeshes();
    this.createCausticTarget();
    await this.compilePipelinesAsync();

    // Size render targets & build screen-dependent bind groups
    this.resize(this.canvas.clientWidth || 1280, this.canvas.clientHeight || 720);
  }

  private createStaticBuffersAndMeshes(): void {
    const d = this.device;

    this.sceneUniformBuffer = d.createBuffer({
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.blockStorageBuffer = d.createBuffer({
      size: this.blockStorageData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.shardStorageBuffer = d.createBuffer({
      size: this.shardStorageData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.splinterStorageBuffer = d.createBuffer({
      size: this.splinterStorageData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.overlaySlotBuffer = d.createBuffer({
      size: this.overlaySlotData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.springUniformBuffer = d.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const mkPostUniform = () =>
      d.createBuffer({
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    this.postUniformExtract = mkPostUniform();
    this.postUniformBlurH = mkPostUniform();
    this.postUniformBlurV = mkPostUniform();
    this.postUniformComposite = mkPostUniform();

    // Upload beveled glass block mesh
    const beveled = createBeveledBlockMesh();
    this.beveledIndexCount = beveled.indexCount;
    this.beveledVertexBuffer = d.createBuffer({
      size: beveled.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    d.queue.writeBuffer(this.beveledVertexBuffer, 0, beveled.vertices.buffer);

    const beveledIdxBytes = Math.ceil(beveled.indices.byteLength / 4) * 4;
    this.beveledIndexBuffer = d.createBuffer({
      size: beveledIdxBytes,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    const paddedBeveledIdx = new Uint8Array(beveledIdxBytes);
    paddedBeveledIdx.set(new Uint8Array(beveled.indices.buffer));
    d.queue.writeBuffer(this.beveledIndexBuffer, 0, paddedBeveledIdx);

    // Upload crisp simple box mesh for caustic pass & top-slot ghost guides
    const box = createSimpleBoxMesh();
    this.boxIndexCount = box.indexCount;
    this.boxVertexBuffer = d.createBuffer({
      size: box.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    d.queue.writeBuffer(this.boxVertexBuffer, 0, box.vertices.buffer);

    this.boxIndexBuffer = d.createBuffer({
      size: box.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    d.queue.writeBuffer(this.boxIndexBuffer, 0, box.indices.buffer);

    // Pre-allocate dynamic shard vertex buffer (65,536 vertices * 40 bytes = 2.62 MB)
    this.shardVertexBuffer = d.createBuffer({
      size: MAX_SHARD_VERTICES * SHARD_VERTEX_FLOATS * 4,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
  }

  private createCausticTarget(): void {
    this.causticTex = this.device.createTexture({
      size: [CAUSTIC_MAP_SIZE, CAUSTIC_MAP_SIZE, 1],
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.causticView = this.causticTex.createView();
  }

  private async compilePipelinesAsync(): Promise<void> {
    const d = this.device;

    const modCaustic = d.createShaderModule({ label: 'CausticPass', code: CAUSTIC_PASS_WGSL });
    const modEnv = d.createShaderModule({ label: 'EnvironmentPass', code: ENVIRONMENT_PASS_WGSL });
    const modTower = d.createShaderModule({ label: 'GlassTowerPass', code: GLASS_TOWER_WGSL });
    const modShard = d.createShaderModule({ label: 'GlassShardPass', code: GLASS_SHARD_WGSL });
    const modOverlay = d.createShaderModule({ label: 'OverlayPass', code: OVERLAY_PASS_WGSL });
    const modPost = d.createShaderModule({ label: 'PostProcessPass', code: POST_PROCESS_WGSL });

    const blockVertexLayout: GPUVertexBufferLayout = {
      arrayStride: 24,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'float32x3' },
      ],
    };

    const shardVertexLayout: GPUVertexBufferLayout = {
      arrayStride: 40,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'float32x3' },
        { shaderLocation: 2, offset: 24, format: 'float32x3' },
        { shaderLocation: 3, offset: 36, format: 'float32' },
      ],
    };

    // Multiplicative optical transmittance blend: out = src * dst
    const multiplicativeBlend: GPUBlendState = {
      color: { srcFactor: 'dst', dstFactor: 'zero', operation: 'add' },
      alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
    };

    const premulAlphaBlend: GPUBlendState = {
      color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    };

    const additiveGlowBlend: GPUBlendState = {
      color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
    };

    const [
      pipeBlockCaustic,
      pipeShardCaustic,
      pipeSky,
      pipeTable,
      pipeGlassTower,
      pipeGlassShard,
      pipeSplinters,
      pipeSlotGuide,
      pipeSpringLine,
      pipeBloomExtract,
      pipeBloomBlur,
      pipeComposite,
    ] = await Promise.all([
      d.createRenderPipelineAsync({
        label: 'pipeBlockCaustic',
        layout: 'auto',
        vertex: { module: modCaustic, entryPoint: 'vs_block_caustic', buffers: [blockVertexLayout] },
        fragment: {
          module: modCaustic,
          entryPoint: 'fs_block_caustic',
          targets: [{ format: 'rgba16float', blend: multiplicativeBlend }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeShardCaustic',
        layout: 'auto',
        vertex: { module: modCaustic, entryPoint: 'vs_shard_caustic', buffers: [shardVertexLayout] },
        fragment: {
          module: modCaustic,
          entryPoint: 'fs_shard_caustic',
          targets: [{ format: 'rgba16float', blend: multiplicativeBlend }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeSky',
        layout: 'auto',
        vertex: { module: modEnv, entryPoint: 'vs_sky' },
        fragment: { module: modEnv, entryPoint: 'fs_sky', targets: [{ format: 'rgba16float' }] },
        depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'less-equal' },
        multisample: { count: 4 },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeTable',
        layout: 'auto',
        vertex: { module: modEnv, entryPoint: 'vs_table' },
        fragment: { module: modEnv, entryPoint: 'fs_table', targets: [{ format: 'rgba16float' }] },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
        multisample: { count: 4 },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeGlassTower',
        layout: 'auto',
        vertex: { module: modTower, entryPoint: 'vs_glass_tower', buffers: [blockVertexLayout] },
        fragment: { module: modTower, entryPoint: 'fs_glass_tower', targets: [{ format: 'rgba16float' }] },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
        multisample: { count: 4 },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeGlassShard',
        layout: 'auto',
        vertex: { module: modShard, entryPoint: 'vs_glass_shard', buffers: [shardVertexLayout] },
        fragment: { module: modShard, entryPoint: 'fs_glass_shard', targets: [{ format: 'rgba16float' }] },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
        multisample: { count: 4 },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeSplinters',
        layout: 'auto',
        vertex: { module: modShard, entryPoint: 'vs_splinter' },
        fragment: {
          module: modShard,
          entryPoint: 'fs_splinter',
          targets: [{ format: 'rgba16float', blend: additiveGlowBlend }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'less' },
        multisample: { count: 4 },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeSlotGuide',
        layout: 'auto',
        vertex: { module: modOverlay, entryPoint: 'vs_slot_guide', buffers: [blockVertexLayout] },
        fragment: {
          module: modOverlay,
          entryPoint: 'fs_slot_guide',
          targets: [{ format: 'rgba16float', blend: premulAlphaBlend }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'less-equal' },
        multisample: { count: 4 },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeSpringLine',
        layout: 'auto',
        vertex: { module: modOverlay, entryPoint: 'vs_spring_line' },
        fragment: {
          module: modOverlay,
          entryPoint: 'fs_spring_line',
          targets: [{ format: 'rgba16float', blend: premulAlphaBlend }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'always' },
        multisample: { count: 4 },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeBloomExtract',
        layout: 'auto',
        vertex: { module: modPost, entryPoint: 'vs_fullscreen' },
        fragment: {
          module: modPost,
          entryPoint: 'fs_bloom_extract',
          targets: [{ format: 'rgba16float' }],
        },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeBloomBlur',
        layout: 'auto',
        vertex: { module: modPost, entryPoint: 'vs_fullscreen' },
        fragment: {
          module: modPost,
          entryPoint: 'fs_bloom_blur',
          targets: [{ format: 'rgba16float' }],
        },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeComposite',
        layout: 'auto',
        vertex: { module: modPost, entryPoint: 'vs_fullscreen' },
        fragment: {
          module: modPost,
          entryPoint: 'fs_composite',
          targets: [{ format: this.canvasFormat }],
        },
      }),
    ]);

    this.pipeBlockCaustic = pipeBlockCaustic;
    this.pipeShardCaustic = pipeShardCaustic;
    this.pipeSky = pipeSky;
    this.pipeTable = pipeTable;
    this.pipeGlassTower = pipeGlassTower;
    this.pipeGlassShard = pipeGlassShard;
    this.pipeSplinters = pipeSplinters;
    this.pipeSlotGuide = pipeSlotGuide;
    this.pipeSpringLine = pipeSpringLine;
    this.pipeBloomExtract = pipeBloomExtract;
    this.pipeBloomBlur = pipeBloomBlur;
    this.pipeComposite = pipeComposite;

    this.buildStaticBindGroups();
  }

  private buildStaticBindGroups(): void {
    const d = this.device;

    this.bgBlockCaustic = d.createBindGroup({
      layout: this.pipeBlockCaustic.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sceneUniformBuffer } },
        { binding: 1, resource: { buffer: this.blockStorageBuffer } },
      ],
    });

    this.bgShardCaustic = d.createBindGroup({
      layout: this.pipeShardCaustic.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sceneUniformBuffer } },
        { binding: 2, resource: { buffer: this.shardStorageBuffer } },
      ],
    });

    this.bgSky = d.createBindGroup({
      layout: this.pipeSky.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.sceneUniformBuffer } }],
    });

    this.bgTable = d.createBindGroup({
      layout: this.pipeTable.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sceneUniformBuffer } },
        { binding: 1, resource: { buffer: this.blockStorageBuffer } },
        { binding: 2, resource: this.causticView },
        { binding: 3, resource: this.linearSampler },
      ],
    });

    this.bgGlassTower = d.createBindGroup({
      layout: this.pipeGlassTower.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sceneUniformBuffer } },
        { binding: 1, resource: { buffer: this.blockStorageBuffer } },
        { binding: 2, resource: this.causticView },
        { binding: 3, resource: this.linearSampler },
      ],
    });

    this.bgSplinters = d.createBindGroup({
      layout: this.pipeSplinters.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sceneUniformBuffer } },
        { binding: 4, resource: { buffer: this.splinterStorageBuffer } },
      ],
    });

    this.bgSlotGuide = d.createBindGroup({
      layout: this.pipeSlotGuide.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sceneUniformBuffer } },
        { binding: 1, resource: { buffer: this.overlaySlotBuffer } },
      ],
    });

    this.bgSpringLine = d.createBindGroup({
      layout: this.pipeSpringLine.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sceneUniformBuffer } },
        { binding: 2, resource: { buffer: this.springUniformBuffer } },
      ],
    });
  }

  public resize(cssWidth: number, cssHeight: number): void {
    if (this.destroyed || !this.device) return;
    // 1.5x DPR combined with native 4x hardware MSAA delivers crisp supersampled glass bevels at locked 60 FPS!
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const w = Math.max(64, Math.floor(cssWidth * dpr));
    const h = Math.max(64, Math.floor(cssHeight * dpr));
    if (w === this.width && h === this.height) return;

    this.width = w;
    this.height = h;
    this.canvas.width = w;
    this.canvas.height = h;

    // Destroy previous dynamic screen textures
    this.hdrColorMSAA?.destroy();
    this.hdrResolveTex?.destroy();
    this.refractionSnapshotTex?.destroy();
    this.depthMSAA?.destroy();
    this.bloomTexA?.destroy();
    this.bloomTexB?.destroy();

    const d = this.device;

    this.hdrColorMSAA = d.createTexture({
      size: [w, h, 1],
      sampleCount: 4,
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.hdrColorMSAAView = this.hdrColorMSAA.createView();

    this.hdrResolveTex = d.createTexture({
      size: [w, h, 1],
      format: 'rgba16float',
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC,
    });
    this.hdrResolveView = this.hdrResolveTex.createView();

    this.refractionSnapshotTex = d.createTexture({
      size: [w, h, 1],
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.refractionSnapshotView = this.refractionSnapshotTex.createView();

    this.depthMSAA = d.createTexture({
      size: [w, h, 1],
      sampleCount: 4,
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.depthMSAAView = this.depthMSAA.createView();

    const bw = Math.max(32, Math.floor(w * 0.5));
    const bh = Math.max(32, Math.floor(h * 0.5));

    this.bloomTexA = d.createTexture({
      size: [bw, bh, 1],
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.bloomViewA = this.bloomTexA.createView();

    this.bloomTexB = d.createTexture({
      size: [bw, bh, 1],
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.bloomViewB = this.bloomTexB.createView();

    // Update post-process uniform buffers once on resize
    d.queue.writeBuffer(this.postUniformExtract, 0, new Float32Array([1 / w, 1 / h, 0.24, 1.0]));
    d.queue.writeBuffer(this.postUniformBlurH, 0, new Float32Array([1 / bw, 0.0, 0.24, 1.0]));
    d.queue.writeBuffer(this.postUniformBlurV, 0, new Float32Array([0.0, 1 / bh, 0.24, 1.0]));
    d.queue.writeBuffer(this.postUniformComposite, 0, new Float32Array([1 / bw, 1 / bh, 0.24, 1.02]));

    // Rebuild screen-size-dependent bind groups
    this.bgGlassShard = d.createBindGroup({
      layout: this.pipeGlassShard.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sceneUniformBuffer } },
        { binding: 1, resource: { buffer: this.shardStorageBuffer } },
        { binding: 2, resource: this.refractionSnapshotView },
        { binding: 3, resource: this.linearSampler },
      ],
    });

    this.bgBloomExtract = d.createBindGroup({
      layout: this.pipeBloomExtract.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.postUniformExtract } },
        { binding: 1, resource: this.hdrResolveView },
        { binding: 2, resource: this.bloomViewB },
        { binding: 3, resource: this.linearSampler },
      ],
    });

    this.bgBloomBlurH = d.createBindGroup({
      layout: this.pipeBloomBlur.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.postUniformBlurH } },
        { binding: 1, resource: this.bloomViewA },
        { binding: 2, resource: this.hdrResolveView },
        { binding: 3, resource: this.linearSampler },
      ],
    });

    this.bgBloomBlurV = d.createBindGroup({
      layout: this.pipeBloomBlur.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.postUniformBlurV } },
        { binding: 1, resource: this.bloomViewB },
        { binding: 2, resource: this.hdrResolveView },
        { binding: 3, resource: this.linearSampler },
      ],
    });

    this.bgComposite = d.createBindGroup({
      layout: this.pipeComposite.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.postUniformComposite } },
        { binding: 1, resource: this.hdrResolveView },
        { binding: 2, resource: this.bloomViewA },
        { binding: 3, resource: this.linearSampler },
      ],
    });
  }

  public render(physics: PhysicsWorld, camera: OrbitCamera, elapsedSec: number): void {
    if (this.destroyed || !this.device || this.width === 0 || this.height === 0) return;

    // 1. Upload any newly fractured Voronoi shard meshes (incremental writeBuffer only on fracture frames)
    const shardAppends = physics.consumePendingShardUploads();
    for (const append of shardAppends) {
      const byteOffset = append.firstVertex * SHARD_VERTEX_FLOATS * 4;
      this.device.queue.writeBuffer(
        this.shardVertexBuffer,
        byteOffset,
        append.vertexData.buffer,
        append.vertexData.byteOffset,
        append.vertexData.byteLength
      );
    }

    // 2. Pack intact stained-glass blocks into blockStorageData
    let intactCount = 0;
    for (let i = 0; i < physics.blocks.length && intactCount < MAX_BLOCKS_GPU; i++) {
      const b = physics.blocks[i];
      if (b.shattered) continue;

      const base = intactCount * BLOCK_FLOATS;
      mat4FromRotationTranslationScale(this.tmpMatA, b.rotation, b.position, [1, 1, 1]);
      mat4FromRigidInverse(this.tmpMatB, b.rotation, b.position);

      this.blockStorageData.set(this.tmpMatA, base + 0);
      this.blockStorageData.set(this.tmpMatB, base + 16);

      const swatch = STAINED_GLASS_PALETTE[b.swatchId];
      this.blockStorageData[base + 32] = swatch.absorption[0];
      this.blockStorageData[base + 33] = swatch.absorption[1];
      this.blockStorageData[base + 34] = swatch.absorption[2];
      this.blockStorageData[base + 35] = b.motif;

      this.blockStorageData[base + 36] = swatch.accentAbsorption[0];
      this.blockStorageData[base + 37] = swatch.accentAbsorption[1];
      this.blockStorageData[base + 38] = swatch.accentAbsorption[2];
      this.blockStorageData[base + 39] = b.rippleSeed;

      this.blockStorageData[base + 40] = b.crackLocalOrigin[0];
      this.blockStorageData[base + 41] = b.crackLocalOrigin[1];
      this.blockStorageData[base + 42] = b.crackLocalOrigin[2];
      this.blockStorageData[base + 43] = b.damage;

      const isHover = physics.hoveredBlockId === b.id ? 1.0 : 0.0;
      const isGrab = physics.activeBlockId === b.id ? 1.0 : 0.0;
      this.blockStorageData[base + 44] = isHover;
      this.blockStorageData[base + 45] = isGrab;
      this.blockStorageData[base + 46] = b.isLegalTarget ? 1.0 : 0.0;
      this.blockStorageData[base + 47] = isGrab > 0.5 ? physics.grabResistance01 : 0.0;

      intactCount++;
    }

    if (intactCount > 0) {
      this.device.queue.writeBuffer(
        this.blockStorageBuffer,
        0,
        this.blockStorageData.buffer,
        0,
        intactCount * BLOCK_FLOATS * 4
      );
    }

    // 3. Pack active 3D Voronoi glass shards
    const shardCount = Math.min(physics.shards.length, MAX_SHARDS_GPU);
    for (let i = 0; i < shardCount; i++) {
      const s = physics.shards[i];
      const base = i * SHARD_FLOATS;
      mat4FromRotationTranslationScale(this.tmpMatA, s.rotation, s.position, [1, 1, 1]);
      this.shardStorageData.set(this.tmpMatA, base + 0);

      const swatch = STAINED_GLASS_PALETTE[s.swatchId];
      this.shardStorageData[base + 16] = swatch.absorption[0];
      this.shardStorageData[base + 17] = swatch.absorption[1];
      this.shardStorageData[base + 18] = swatch.absorption[2];
      this.shardStorageData[base + 19] = s.motif;

      this.shardStorageData[base + 20] = swatch.accentAbsorption[0];
      this.shardStorageData[base + 21] = swatch.accentAbsorption[1];
      this.shardStorageData[base + 22] = swatch.accentAbsorption[2];
      this.shardStorageData[base + 23] = s.rippleSeed;
    }
    if (shardCount > 0) {
      this.device.queue.writeBuffer(
        this.shardStorageBuffer,
        0,
        this.shardStorageData.buffer,
        0,
        shardCount * SHARD_FLOATS * 4
      );
    }

    // 4. Pack glittering micro-splinters + 48 sunlit cathedral dust motes floating in sunbeams!
    let particleCount = 0;
    for (let i = 0; i < physics.splinters.length && particleCount < MAX_SPLINTERS_GPU; i++) {
      const sp = physics.splinters[i];
      const base = particleCount * SPLINTER_FLOATS;
      const fade = 1.0 - Math.pow(sp.life / sp.maxLife, 2);
      this.splinterStorageData[base + 0] = sp.pos[0];
      this.splinterStorageData[base + 1] = sp.pos[1];
      this.splinterStorageData[base + 2] = sp.pos[2];
      this.splinterStorageData[base + 3] = sp.scale * fade;

      const sw = STAINED_GLASS_PALETTE[sp.swatchId];
      this.splinterStorageData[base + 4] = Math.exp(-sw.absorption[0] * 0.12) * 1.4;
      this.splinterStorageData[base + 5] = Math.exp(-sw.absorption[1] * 0.12) * 1.4;
      this.splinterStorageData[base + 6] = Math.exp(-sw.absorption[2] * 0.12) * 1.4;
      this.splinterStorageData[base + 7] = sp.rot;
      particleCount++;
    }

    // Ambient sunlit cathedral dust motes gently drifting around the stained-glass tower
    const moteCount = 44;
    for (let m = 0; m < moteCount && particleCount < MAX_SPLINTERS_GPU; m++) {
      const base = particleCount * SPLINTER_FLOATS;
      const seed = m * 3.719;
      const r = 0.65 + (m % 7) * 0.42;
      const theta = seed + elapsedSec * (0.04 + (m % 5) * 0.012);
      const y = 0.35 + ((seed * 0.41 + elapsedSec * 0.05) % 3.6);
      this.splinterStorageData[base + 0] = Math.cos(theta) * r;
      this.splinterStorageData[base + 1] = y;
      this.splinterStorageData[base + 2] = Math.sin(theta) * r;
      this.splinterStorageData[base + 3] = 0.016 + (m % 3) * 0.006;
      this.splinterStorageData[base + 4] = this.lightingPreset.sunColor[0] * 0.38;
      this.splinterStorageData[base + 5] = this.lightingPreset.sunColor[1] * 0.36;
      this.splinterStorageData[base + 6] = this.lightingPreset.sunColor[2] * 0.32;
      this.splinterStorageData[base + 7] = -100.0; // Dust flag
      particleCount++;
    }
    this.device.queue.writeBuffer(
      this.splinterStorageBuffer,
      0,
      this.splinterStorageData.buffer,
      0,
      particleCount * SPLINTER_FLOATS * 4
    );

    // 5. Pack spatial overlays (Top Placement Slot Guides & Spring Pull Filament)
    let slotDrawCount = 0;
    if (physics.turnPhase === 'EXTRACTED') {
      const carriedBlk = physics.activeBlockId !== null ? physics.blocks[physics.activeBlockId] : null;
      const sw = carriedBlk ? STAINED_GLASS_PALETTE[carriedBlk.swatchId] : STAINED_GLASS_PALETTE[2];
      const tintR = Math.max(0.45, Math.exp(-sw.absorption[0] * 0.14));
      const tintG = Math.max(0.45, Math.exp(-sw.absorption[1] * 0.14));
      const tintB = Math.max(0.45, Math.exp(-sw.absorption[2] * 0.14));

      for (let i = 0; i < physics.openTopSlots.length && slotDrawCount < 6; i++) {
        const slot = physics.openTopSlots[i];
        const base = slotDrawCount * 20;
        mat4FromRotationTranslationScale(this.tmpMatA, slot.rotation, slot.position, [0.992, 0.96, 0.985]);
        this.overlaySlotData.set(this.tmpMatA, base + 0);
        this.overlaySlotData[base + 16] = tintR;
        this.overlaySlotData[base + 17] = tintG;
        this.overlaySlotData[base + 18] = tintB;
        this.overlaySlotData[base + 19] = physics.hoveredSlotIndex === slot.slotIndex ? 1.0 : 0.0;
        slotDrawCount++;
      }
      if (slotDrawCount > 0) {
        this.device.queue.writeBuffer(
          this.overlaySlotBuffer,
          0,
          this.overlaySlotData.buffer,
          0,
          slotDrawCount * 20 * 4
        );
      }
    }

    const showSpring =
      physics.isPointerDraggingBlock &&
      physics.activeBlockId !== null &&
      (physics.turnPhase === 'PULLING' || physics.gameMode === 'sandbox');
    if (showSpring) {
      this.springUniformData[0] = physics.grabCurrentWorldOnBlock[0];
      this.springUniformData[1] = physics.grabCurrentWorldOnBlock[1];
      this.springUniformData[2] = physics.grabCurrentWorldOnBlock[2];
      this.springUniformData[3] = 1.0;
      this.springUniformData[4] = physics.grabWorldAnchor[0];
      this.springUniformData[5] = physics.grabWorldAnchor[1];
      this.springUniformData[6] = physics.grabWorldAnchor[2];
      this.springUniformData[7] = physics.grabResistance01;
    } else {
      this.springUniformData[3] = 0.0;
    }
    this.device.queue.writeBuffer(this.springUniformBuffer, 0, this.springUniformData.buffer);

    // 6. Compute Sun Orthographic View-Projection & Upload SceneUniforms
    const az = (this.lightingPreset.sunAzimuthDeg * Math.PI) / 180;
    const el = (this.lightingPreset.sunElevationDeg * Math.PI) / 180;
    const sunDir: Vec3 = [
      Math.cos(el) * Math.sin(az),
      Math.sin(el),
      Math.cos(el) * Math.cos(az),
    ];

    const sunTargetY = 1.25;
    const sunEye: Vec3 = [sunDir[0] * 10.0, sunTargetY + sunDir[1] * 10.0, sunDir[2] * 10.0];
    mat4LookAt(this.sunViewMatrix, sunEye, [0, sunTargetY, 0], [0, 1, 0]);
    mat4OrthoZO(this.sunProjMatrix, -4.85, 4.85, -4.85, 4.85, 1.0, 22.0);
    mat4Multiply(this.sunViewProjMatrix, this.sunProjMatrix, this.sunViewMatrix);

    this.sceneUniformData.set(camera.viewProjMatrix, 0);
    this.sceneUniformData.set(camera.invViewProjMatrix, 16);
    this.sceneUniformData.set(this.sunViewProjMatrix, 32);

    this.sceneUniformData[48] = camera.position[0];
    this.sceneUniformData[49] = camera.position[1];
    this.sceneUniformData[50] = camera.position[2];
    this.sceneUniformData[51] = elapsedSec;

    this.sceneUniformData[52] = sunDir[0];
    this.sceneUniformData[53] = sunDir[1];
    this.sceneUniformData[54] = sunDir[2];
    this.sceneUniformData[55] = intactCount;

    this.sceneUniformData[56] = this.lightingPreset.sunColor[0];
    this.sceneUniformData[57] = this.lightingPreset.sunColor[1];
    this.sceneUniformData[58] = this.lightingPreset.sunColor[2];
    this.sceneUniformData[59] = this.width / Math.max(1, this.height);

    this.sceneUniformData[60] = this.lightingPreset.skyTint[0];
    this.sceneUniformData[61] = this.lightingPreset.skyTint[1];
    this.sceneUniformData[62] = this.lightingPreset.skyTint[2];
    this.sceneUniformData[63] = 1.0;

    this.device.queue.writeBuffer(this.sceneUniformBuffer, 0, this.sceneUniformData.buffer);

    // 7. Record all GPU Passes into ONE single CommandEncoder!
    const swapchainView = this.context.getCurrentTexture().createView();
    const encoder = this.device.createCommandEncoder({ label: 'VitrailFrameEncoder' });

    // Pass 1: Sun Colored Stained-Glass Caustic Transmission Map (clears to pure white 1.0)
    {
      const pass = encoder.beginRenderPass({
        label: 'CausticTransmissionPass',
        colorAttachments: [
          {
            view: this.causticView,
            clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });

      if (intactCount > 0) {
        pass.setPipeline(this.pipeBlockCaustic);
        pass.setBindGroup(0, this.bgBlockCaustic);
        pass.setVertexBuffer(0, this.boxVertexBuffer);
        pass.setIndexBuffer(this.boxIndexBuffer, 'uint16');
        pass.drawIndexed(this.boxIndexCount, intactCount, 0, 0, 0);
      }

      if (shardCount > 0) {
        pass.setPipeline(this.pipeShardCaustic);
        pass.setBindGroup(0, this.bgShardCaustic);
        pass.setVertexBuffer(0, this.shardVertexBuffer);
        for (let i = 0; i < shardCount; i++) {
          const sh = physics.shards[i];
          pass.draw(sh.vertexCount, 1, sh.firstVertex, i);
        }
      }
      pass.end();
    }

    // Pass 2 & 3: Cathedral Sky + Caustic Oak Table + Multi-Block Raytraced Stained-Glass Tower (4x MSAA)
    {
      const pass = encoder.beginRenderPass({
        label: 'MainOpaqueAndTowerPass',
        colorAttachments: [
          {
            view: this.hdrColorMSAAView,
            resolveTarget: this.hdrResolveView,
            clearValue: { r: 0.05, g: 0.05, b: 0.06, a: 1.0 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
        depthStencilAttachment: {
          view: this.depthMSAAView,
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      });

      // Sky Dome
      pass.setPipeline(this.pipeSky);
      pass.setBindGroup(0, this.bgSky);
      pass.draw(3, 1, 0, 0);

      // Quarter-sawn Oak Pedestal Table with Stained-Glass Caustics
      pass.setPipeline(this.pipeTable);
      pass.setBindGroup(0, this.bgTable);
      pass.draw(96 * 9, 1, 0, 0);

      // Intact Stained-Glass Jenga Tower
      if (intactCount > 0) {
        pass.setPipeline(this.pipeGlassTower);
        pass.setBindGroup(0, this.bgGlassTower);
        pass.setVertexBuffer(0, this.beveledVertexBuffer);
        pass.setIndexBuffer(this.beveledIndexBuffer, 'uint16');
        pass.drawIndexed(this.beveledIndexCount, intactCount, 0, 0, 0);
      }
      pass.end();
    }

    // Copy resolved HDR scene (Table + Tower) into refractionSnapshotTex for 3D Glass Shard refraction!
    encoder.copyTextureToTexture(
      { texture: this.hdrResolveTex },
      { texture: this.refractionSnapshotTex },
      [this.width, this.height, 1]
    );

    // Pass 4: Refractive 3D Voronoi Glass Shards + Micro-Splinters + Spatial Placement Guides (4x MSAA)
    {
      const pass = encoder.beginRenderPass({
        label: 'ShardsAndOverlaysPass',
        colorAttachments: [
          {
            view: this.hdrColorMSAAView,
            resolveTarget: this.hdrResolveView,
            loadOp: 'load',
            storeOp: 'discard',
          },
        ],
        depthStencilAttachment: {
          view: this.depthMSAAView,
          depthLoadOp: 'load',
          depthStoreOp: 'discard',
        },
      });

      if (shardCount > 0) {
        pass.setPipeline(this.pipeGlassShard);
        pass.setBindGroup(0, this.bgGlassShard);
        pass.setVertexBuffer(0, this.shardVertexBuffer);
        for (let i = 0; i < shardCount; i++) {
          const sh = physics.shards[i];
          pass.draw(sh.vertexCount, 1, sh.firstVertex, i);
        }
      }

      if (particleCount > 0) {
        pass.setPipeline(this.pipeSplinters);
        pass.setBindGroup(0, this.bgSplinters);
        pass.draw(6, particleCount, 0, 0);
      }

      if (slotDrawCount > 0) {
        pass.setPipeline(this.pipeSlotGuide);
        pass.setBindGroup(0, this.bgSlotGuide);
        pass.setVertexBuffer(0, this.boxVertexBuffer);
        pass.setIndexBuffer(this.boxIndexBuffer, 'uint16');
        pass.drawIndexed(this.boxIndexCount, slotDrawCount, 0, 0, 0);
      }

      if (showSpring) {
        pass.setPipeline(this.pipeSpringLine);
        pass.setBindGroup(0, this.bgSpringLine);
        pass.draw(6, 1, 0, 0);
      }

      pass.end();
    }

    // Pass 5a: HDR Bloom Bright-Pass Extract
    {
      const pass = encoder.beginRenderPass({
        label: 'BloomExtractPass',
        colorAttachments: [
          { view: this.bloomViewA, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } },
        ],
      });
      pass.setPipeline(this.pipeBloomExtract);
      pass.setBindGroup(0, this.bgBloomExtract);
      pass.draw(3, 1, 0, 0);
      pass.end();
    }

    // Pass 5b: Separable Gaussian Blur H -> V
    {
      const passH = encoder.beginRenderPass({
        label: 'BloomBlurHPass',
        colorAttachments: [
          { view: this.bloomViewB, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } },
        ],
      });
      passH.setPipeline(this.pipeBloomBlur);
      passH.setBindGroup(0, this.bgBloomBlurH);
      passH.draw(3, 1, 0, 0);
      passH.end();

      const passV = encoder.beginRenderPass({
        label: 'BloomBlurVPass',
        colorAttachments: [
          { view: this.bloomViewA, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } },
        ],
      });
      passV.setPipeline(this.pipeBloomBlur);
      passV.setBindGroup(0, this.bgBloomBlurV);
      passV.draw(3, 1, 0, 0);
      passV.end();
    }

    // Pass 5c: ACES Filmic Tone Mapping + Optical Bloom Composite -> Swapchain
    {
      const pass = encoder.beginRenderPass({
        label: 'CompositePass',
        colorAttachments: [
          {
            view: swapchainView,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });
      pass.setPipeline(this.pipeComposite);
      pass.setBindGroup(0, this.bgComposite);
      pass.draw(3, 1, 0, 0);
      pass.end();
    }

    this.device.queue.submit([encoder.finish()]);
  }

  public dispose(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    try {
      this.context?.unconfigure();
      this.causticTex?.destroy();
      this.hdrColorMSAA?.destroy();
      this.hdrResolveTex?.destroy();
      this.refractionSnapshotTex?.destroy();
      this.depthMSAA?.destroy();
      this.bloomTexA?.destroy();
      this.bloomTexB?.destroy();
      this.device?.destroy();
    } catch {
      // Ignore cleanup errors on already-lost GPU device
    }
  }
}
