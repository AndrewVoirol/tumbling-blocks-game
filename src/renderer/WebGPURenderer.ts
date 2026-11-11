import {
  createBeveledBlockMesh,
  createCausticRefractionMesh,
  createSimpleBoxMesh,
} from '../geometry/beveledBlockMesh';
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

const SUN_DEPTH_MAP_SIZE = 2048;
const CAUSTIC_MAP_SIZE = 1024;
const MAX_BLOCKS_GPU = 96;
const BLOCK_FLOATS = 52; // 208 bytes per GPUBlockData (2 mat4 + 5 vec4)
const MAX_SHARDS_GPU = 512;
const SHARD_FLOATS = 76; // 304 bytes per GPUShardInstance (mat4 + 3 vec4 + 12 convex planes)
const MAX_SHARD_VERTICES = 67584; // 512 slots * 132 vertices/slot
const SHARD_VERTEX_FLOATS = 10; // 40 bytes per shard vertex
const MAX_SPLINTERS_GPU = 320; // Grounded glass cullet chips
const SPLINTER_FLOATS = 8; // 32 bytes per GPUSplinter

export class WebGPURenderer {
  public device!: GPUDevice;
  public context!: GPUCanvasContext;
  public canvasFormat!: GPUTextureFormat;
  public preferredSdrFormat!: GPUTextureFormat;
  public isExtendedHdr: boolean = false;
  private canvas: HTMLCanvasElement;
  private hdrMediaQuery: MediaQueryList | null = null;
  private onHdrMediaChange: ((e: MediaQueryListEvent) => void) | null = null;

  // Static & Pre-allocated GPU Buffers
  private sceneUniformBuffer!: GPUBuffer;
  private blockCentersUniformBuffer!: GPUBuffer;
  private blockStorageBuffer!: GPUBuffer;
  private shardStorageBuffer!: GPUBuffer;
  private splinterStorageBuffer!: GPUBuffer;
  private overlaySlotBuffer!: GPUBuffer;
  private springUniformBuffer!: GPUBuffer;

  private postUniformExtract!: GPUBuffer;
  private postUniformBlurH!: GPUBuffer;
  private postUniformBlurV!: GPUBuffer;
  private postUniformDownsample!: GPUBuffer;
  private postUniformWideBlurH!: GPUBuffer;
  private postUniformWideBlurV!: GPUBuffer;
  private postUniformComposite!: GPUBuffer;

  private beveledVertexBuffer!: GPUBuffer;
  private beveledIndexBuffer!: GPUBuffer;
  private beveledIndexCount: number = 0;

  private causticRefractVertexBuffer!: GPUBuffer;
  private causticRefractIndexBuffer!: GPUBuffer;
  private causticRefractIndexCount: number = 0;

  private boxVertexBuffer!: GPUBuffer;
  private boxIndexBuffer!: GPUBuffer;
  private boxIndexCount: number = 0;

  private shardVertexBuffer!: GPUBuffer;

  // Render Targets
  private sunDepthTex!: GPUTexture;
  private sunDepthView!: GPUTextureView;
  private sunOpticalDepthTex!: GPUTexture;
  private sunOpticalDepthView!: GPUTextureView;
  private causticTex!: GPUTexture;
  private causticView!: GPUTextureView;

  private hdrResolveTex!: GPUTexture;
  private hdrResolveView!: GPUTextureView;
  private refractionSnapshotTex!: GPUTexture;
  private refractionSnapshotView!: GPUTextureView;
  private depthResolveTex!: GPUTexture;
  private depthResolveView!: GPUTextureView;

  // Dual-octave optical bloom targets (1/2 res tight glare + 1/6 res wide atmospheric halation)
  private bloomTexA!: GPUTexture;
  private bloomViewA!: GPUTextureView;
  private bloomTexB!: GPUTexture;
  private bloomViewB!: GPUTextureView;
  private bloomWideTexA!: GPUTexture;
  private bloomWideViewA!: GPUTextureView;
  private bloomWideTexB!: GPUTexture;
  private bloomWideViewB!: GPUTextureView;

  private linearSampler!: GPUSampler;

  // Render Pipelines
  private pipeSunDepthBlock!: GPURenderPipeline;
  private pipeSunDepthShard!: GPURenderPipeline;
  private pipeOpticalDepthBlock!: GPURenderPipeline;
  private pipeOpticalDepthShard!: GPURenderPipeline;
  private pipeBlockCaustic!: GPURenderPipeline;
  private pipeShardCaustic!: GPURenderPipeline;
  private pipeSky!: GPURenderPipeline;
  private pipeTable!: GPURenderPipeline;
  private pipeTowerDepth!: GPURenderPipeline;
  private pipeGlassTower!: GPURenderPipeline;
  private pipeGlassShard!: GPURenderPipeline;
  private pipeSplinters!: GPURenderPipeline;
  private pipeSlotGuide!: GPURenderPipeline;
  private pipeSpringLine!: GPURenderPipeline;
  private pipeBloomExtract!: GPURenderPipeline;
  private pipeBloomDownsample!: GPURenderPipeline;
  private pipeBloomBlur!: GPURenderPipeline;
  private bglComposite!: GPUBindGroupLayout;
  private pipeCompositeSdr!: GPURenderPipeline;
  private pipeCompositeHdr!: GPURenderPipeline;
  private pipeComposite!: GPURenderPipeline;

  // Pre-allocated Bind Groups (zero allocation inside rAF!)
  private bgSunDepthBlock!: GPUBindGroup;
  private bgSunDepthShard!: GPUBindGroup;
  private bgOpticalDepthBlock!: GPUBindGroup;
  private bgOpticalDepthShard!: GPUBindGroup;
  private bgBlockCaustic!: GPUBindGroup;
  private bgShardCaustic!: GPUBindGroup;
  private bgSky!: GPUBindGroup;
  private bgTable!: GPUBindGroup;
  private bgTowerDepth!: GPUBindGroup;
  private bgGlassTower!: GPUBindGroup;
  private bgGlassShard!: GPUBindGroup;
  private bgSplinters!: GPUBindGroup;
  private bgSlotGuide!: GPUBindGroup;
  private bgSpringLine!: GPUBindGroup;
  private bgBloomExtract!: GPUBindGroup;
  private bgBloomBlurH!: GPUBindGroup;
  private bgBloomBlurV!: GPUBindGroup;
  private bgBloomDownsample!: GPUBindGroup;
  private bgBloomWideBlurH!: GPUBindGroup;
  private bgBloomWideBlurV!: GPUBindGroup;
  private bgComposite!: GPUBindGroup;

  // Pre-allocated CPU Float32Array / Uint32Array / Int32Array staging views
  private sceneUniformData = new Float32Array(72);
  private blockCentersData = new Float32Array(98 * 4);
  private blockStorageData = new Float32Array(MAX_BLOCKS_GPU * BLOCK_FLOATS);
  private shardStorageData = new Float32Array(MAX_SHARDS_GPU * SHARD_FLOATS);
  private shardStorageU32 = new Uint32Array(this.shardStorageData.buffer);
  private splinterStorageData = new Float32Array(MAX_SPLINTERS_GPU * SPLINTER_FLOATS);
  private overlaySlotData = new Float32Array(8 * 20);
  private springUniformData = new Float32Array(8);
  private compositeUniformData = new Float32Array(8);

  private blockSortOrder = new Int32Array(MAX_BLOCKS_GPU);
  private blockSortDepths = new Float32Array(MAX_BLOCKS_GPU);
  private shardSortOrder = new Int32Array(MAX_SHARDS_GPU);
  private shardSortDepths = new Float32Array(MAX_SHARDS_GPU);

  private sunViewMatrix = mat4Create();
  private sunProjMatrix = mat4Create();
  private sunViewProjMatrix = mat4Create();
  private tmpMatA = mat4Create();
  private tmpMatB = mat4Create();

  public lightingPreset: LightingPreset = LIGHTING_PRESETS[0];
  public lightingOverride: Partial<LightingPreset> | null = null;
  public referenceCardLuminance: number = 0;
  public benchHideBlocks: boolean = false;
  public polariscopeMode: boolean = false;
  private width: number = 0;
  private height: number = 0;
  private bloomW: number = 32;
  private bloomH: number = 32;
  private destroyed: boolean = false;

  // Optional GPU hardware timestamp query telemetry (Apple Silicon Metal-3)
  public supportsTimestamps: boolean = false;
  public lastGpuFrameMs: number = 0;
  private timestampQuerySet: GPUQuerySet | null = null;
  private timestampResolveBuffer: GPUBuffer | null = null;
  private timestampReadBuffer: GPUBuffer | null = null;
  private timestampReadPending: boolean = false;
  private tsWritesStart: GPURenderPassTimestampWrites | undefined = undefined;
  private tsWritesEnd: GPURenderPassTimestampWrites | undefined = undefined;

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

    this.supportsTimestamps = adapter.features.has('timestamp-query');
    this.device = await adapter.requestDevice(
      this.supportsTimestamps ? { requiredFeatures: ['timestamp-query'] } : undefined
    );
    const ctx = this.canvas.getContext('webgpu');
    if (!ctx) {
      throw new Error('Unable to configure WebGPU canvas context.');
    }
    this.context = ctx;
    this.preferredSdrFormat = navigator.gpu.getPreferredCanvasFormat();
    const hdrQuery =
      typeof window !== 'undefined' && typeof window.matchMedia === 'function'
        ? window.matchMedia('(dynamic-range: high)')
        : null;
    this.configureSwapchain(Boolean(hdrQuery?.matches));
    if (hdrQuery) {
      this.hdrMediaQuery = hdrQuery;
      this.onHdrMediaChange = (e: MediaQueryListEvent) => {
        if (!this.destroyed) {
          this.setExtendedHdr(e.matches);
        }
      };
      hdrQuery.addEventListener('change', this.onHdrMediaChange);
    }

    if (this.supportsTimestamps) {
      this.timestampQuerySet = this.device.createQuerySet({
        type: 'timestamp',
        count: 2,
      });
      this.timestampResolveBuffer = this.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.timestampReadBuffer = this.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      this.tsWritesStart = {
        querySet: this.timestampQuerySet,
        beginningOfPassWriteIndex: 0,
      };
      this.tsWritesEnd = {
        querySet: this.timestampQuerySet,
        endOfPassWriteIndex: 1,
      };
    }

    this.linearSampler = this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    this.createStaticBuffersAndMeshes();
    this.createCausticAndShadowTargets();
    await this.compilePipelinesAsync();

    this.resize(this.canvas.clientWidth || 1280, this.canvas.clientHeight || 720);
  }

  private configureSwapchain(preferExtendedHdr: boolean): void {
    if (preferExtendedHdr) {
      try {
        this.context.configure({
          device: this.device,
          format: 'rgba16float',
          alphaMode: 'opaque',
          toneMapping: { mode: 'extended' },
        });
        this.canvasFormat = 'rgba16float';
        this.isExtendedHdr = true;
        if (this.pipeCompositeHdr) {
          this.pipeComposite = this.pipeCompositeHdr;
        }
        return;
      } catch {
        // Fall through to standard SDR swapchain when extended rgba16float is unsupported
      }
    }

    try {
      this.context.configure({
        device: this.device,
        format: this.preferredSdrFormat,
        alphaMode: 'opaque',
        toneMapping: { mode: 'standard' },
      });
    } catch {
      this.context.configure({
        device: this.device,
        format: this.preferredSdrFormat,
        alphaMode: 'opaque',
      });
    }
    this.canvasFormat = this.preferredSdrFormat;
    this.isExtendedHdr = false;
    if (this.pipeCompositeSdr) {
      this.pipeComposite = this.pipeCompositeSdr;
    }
  }

  public setExtendedHdr(enabled: boolean): void {
    if (this.destroyed || !this.context || !this.device) return;
    this.configureSwapchain(enabled);
  }

  private createStaticBuffersAndMeshes(): void {
    const d = this.device;

    this.sceneUniformBuffer = d.createBuffer({
      size: this.sceneUniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.blockCentersUniformBuffer = d.createBuffer({
      size: this.blockCentersData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
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
    this.postUniformDownsample = mkPostUniform();
    this.postUniformWideBlurH = mkPostUniform();
    this.postUniformWideBlurV = mkPostUniform();
    this.postUniformComposite = mkPostUniform();

    // Upload 1.25 mm diamond-polished beveled glass block mesh (analytical rounded normals in shader)
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

    // Upload 13x13 per-face tessellated waterglass & 1.25 mm bevel proxy for Snell caustic refraction
    const causticRefract = createCausticRefractionMesh();
    this.causticRefractIndexCount = causticRefract.indexCount;
    this.causticRefractVertexBuffer = d.createBuffer({
      size: causticRefract.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    d.queue.writeBuffer(this.causticRefractVertexBuffer, 0, causticRefract.vertices.buffer);

    const causticRefractIdxBytes = Math.ceil(causticRefract.indices.byteLength / 4) * 4;
    this.causticRefractIndexBuffer = d.createBuffer({
      size: causticRefractIdxBytes,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    const paddedCausticIdx = new Uint8Array(causticRefractIdxBytes);
    paddedCausticIdx.set(new Uint8Array(causticRefract.indices.buffer));
    d.queue.writeBuffer(this.causticRefractIndexBuffer, 0, paddedCausticIdx);

    // Upload simple box mesh for top-slot placement ghost guides
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

    // Pre-allocate dynamic shard vertex buffer (67,584 vertices * 40 bytes = 2.70 MB) with STORAGE pull
    this.shardVertexBuffer = d.createBuffer({
      size: MAX_SHARD_VERTICES * SHARD_VERTEX_FLOATS * 4,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  private createCausticAndShadowTargets(): void {
    this.sunDepthTex = this.device.createTexture({
      size: [SUN_DEPTH_MAP_SIZE, SUN_DEPTH_MAP_SIZE, 1],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.sunDepthView = this.sunDepthTex.createView();

    this.sunOpticalDepthTex = this.device.createTexture({
      size: [CAUSTIC_MAP_SIZE, CAUSTIC_MAP_SIZE, 1],
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.sunOpticalDepthView = this.sunOpticalDepthTex.createView();

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

    // Spectral optical depth accumulation (.rgb = sum tau_sun) + lowest caster height (.a = min exitWorldY)
    const opticalDepthMinYBlend: GPUBlendState = {
      color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'min' },
    };

    // Additive Snell-refracted Jacobian caustic flux: out = src + dst
    const additiveFluxBlend: GPUBlendState = {
      color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
    };

    // Premultiplied RGB overlay blend that preserves destination alpha (linear camera distance for DoF!)
    const premulPreserveDstAlphaBlend: GPUBlendState = {
      color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
    };

    this.bglComposite = d.createBindGroupLayout({
      label: 'bglComposite',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      ],
    });
    const plComposite = d.createPipelineLayout({
      label: 'plComposite',
      bindGroupLayouts: [this.bglComposite],
    });

    const [
      pipeSunDepthBlock,
      pipeSunDepthShard,
      pipeOpticalDepthBlock,
      pipeOpticalDepthShard,
      pipeBlockCaustic,
      pipeShardCaustic,
      pipeSky,
      pipeTable,
      pipeTowerDepth,
      pipeGlassTower,
      pipeGlassShard,
      pipeSplinters,
      pipeSlotGuide,
      pipeSpringLine,
      pipeBloomExtract,
      pipeBloomDownsample,
      pipeBloomBlur,
      pipeCompositeSdr,
      pipeCompositeHdr,
    ] = await Promise.all([
      d.createRenderPipelineAsync({
        label: 'pipeSunDepthBlock',
        layout: 'auto',
        vertex: { module: modCaustic, entryPoint: 'vs_sun_depth_block', buffers: [blockVertexLayout] },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeSunDepthShard',
        layout: 'auto',
        vertex: { module: modCaustic, entryPoint: 'vs_sun_depth_shard' },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeOpticalDepthBlock',
        layout: 'auto',
        vertex: { module: modCaustic, entryPoint: 'vs_optical_depth_block', buffers: [blockVertexLayout] },
        fragment: {
          module: modCaustic,
          entryPoint: 'fs_optical_depth_block',
          targets: [{ format: 'rgba16float', blend: opticalDepthMinYBlend }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeOpticalDepthShard',
        layout: 'auto',
        vertex: { module: modCaustic, entryPoint: 'vs_optical_depth_shard' },
        fragment: {
          module: modCaustic,
          entryPoint: 'fs_optical_depth_shard',
          targets: [{ format: 'rgba16float', blend: opticalDepthMinYBlend }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeBlockCaustic',
        layout: 'auto',
        vertex: { module: modCaustic, entryPoint: 'vs_block_caustic_flux', buffers: [blockVertexLayout] },
        fragment: {
          module: modCaustic,
          entryPoint: 'fs_block_caustic_flux',
          targets: [{ format: 'rgba16float', blend: additiveFluxBlend }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeShardCaustic',
        layout: 'auto',
        vertex: { module: modCaustic, entryPoint: 'vs_shard_caustic' },
        fragment: {
          module: modCaustic,
          entryPoint: 'fs_shard_caustic',
          targets: [{ format: 'rgba16float', blend: additiveFluxBlend }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeSky',
        layout: 'auto',
        vertex: { module: modEnv, entryPoint: 'vs_sky' },
        fragment: { module: modEnv, entryPoint: 'fs_sky', targets: [{ format: 'rgba16float' }] },
        depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'less-equal' },
        multisample: { count: 1 },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeTable',
        layout: 'auto',
        vertex: { module: modEnv, entryPoint: 'vs_table' },
        fragment: { module: modEnv, entryPoint: 'fs_table', targets: [{ format: 'rgba16float' }] },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
        multisample: { count: 1 },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeTowerDepth',
        layout: 'auto',
        vertex: { module: modTower, entryPoint: 'vs_tower_depth', buffers: [blockVertexLayout] },
        fragment: {
          module: modTower,
          entryPoint: 'fs_tower_depth',
          targets: [{ format: 'rgba16float', writeMask: 0 }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
        multisample: { count: 1 },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeGlassTower',
        layout: 'auto',
        vertex: { module: modTower, entryPoint: 'vs_glass_tower', buffers: [blockVertexLayout] },
        fragment: { module: modTower, entryPoint: 'fs_glass_tower', targets: [{ format: 'rgba16float' }] },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'less-equal' },
        multisample: { count: 1 },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeGlassShard',
        layout: 'auto',
        vertex: { module: modShard, entryPoint: 'vs_glass_shard' },
        fragment: { module: modShard, entryPoint: 'fs_glass_shard', targets: [{ format: 'rgba16float' }] },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
        multisample: { count: 1 },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeSplinters',
        layout: 'auto',
        vertex: { module: modShard, entryPoint: 'vs_splinter' },
        fragment: {
          module: modShard,
          entryPoint: 'fs_splinter',
          targets: [{ format: 'rgba16float', blend: premulPreserveDstAlphaBlend }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'less-equal' },
        multisample: { count: 1 },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeSlotGuide',
        layout: 'auto',
        vertex: { module: modOverlay, entryPoint: 'vs_slot_guide', buffers: [blockVertexLayout] },
        fragment: {
          module: modOverlay,
          entryPoint: 'fs_slot_guide',
          targets: [{ format: 'rgba16float', blend: premulPreserveDstAlphaBlend }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'less-equal' },
        multisample: { count: 1 },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeSpringLine',
        layout: 'auto',
        vertex: { module: modOverlay, entryPoint: 'vs_spring_line' },
        fragment: {
          module: modOverlay,
          entryPoint: 'fs_spring_line',
          targets: [{ format: 'rgba16float', blend: premulPreserveDstAlphaBlend }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'always' },
        multisample: { count: 1 },
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
        label: 'pipeBloomDownsample',
        layout: 'auto',
        vertex: { module: modPost, entryPoint: 'vs_fullscreen' },
        fragment: {
          module: modPost,
          entryPoint: 'fs_bloom_downsample',
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
        label: 'pipeCompositeSdr',
        layout: plComposite,
        vertex: { module: modPost, entryPoint: 'vs_fullscreen' },
        fragment: {
          module: modPost,
          entryPoint: 'fs_composite',
          constants: { HDR_EXTENDED: 0 },
          targets: [{ format: this.preferredSdrFormat }],
        },
      }),
      d.createRenderPipelineAsync({
        label: 'pipeCompositeHdr',
        layout: plComposite,
        vertex: { module: modPost, entryPoint: 'vs_fullscreen' },
        fragment: {
          module: modPost,
          entryPoint: 'fs_composite',
          constants: { HDR_EXTENDED: 1 },
          targets: [{ format: 'rgba16float' }],
        },
      }),
    ]);

    this.pipeSunDepthBlock = pipeSunDepthBlock;
    this.pipeSunDepthShard = pipeSunDepthShard;
    this.pipeOpticalDepthBlock = pipeOpticalDepthBlock;
    this.pipeOpticalDepthShard = pipeOpticalDepthShard;
    this.pipeBlockCaustic = pipeBlockCaustic;
    this.pipeShardCaustic = pipeShardCaustic;
    this.pipeSky = pipeSky;
    this.pipeTable = pipeTable;
    this.pipeTowerDepth = pipeTowerDepth;
    this.pipeGlassTower = pipeGlassTower;
    this.pipeGlassShard = pipeGlassShard;
    this.pipeSplinters = pipeSplinters;
    this.pipeSlotGuide = pipeSlotGuide;
    this.pipeSpringLine = pipeSpringLine;
    this.pipeBloomExtract = pipeBloomExtract;
    this.pipeBloomDownsample = pipeBloomDownsample;
    this.pipeBloomBlur = pipeBloomBlur;
    this.pipeCompositeSdr = pipeCompositeSdr;
    this.pipeCompositeHdr = pipeCompositeHdr;
    this.pipeComposite = this.isExtendedHdr ? pipeCompositeHdr : pipeCompositeSdr;

    this.buildStaticBindGroups();
  }

  private buildStaticBindGroups(): void {
    const d = this.device;

    this.bgSunDepthBlock = d.createBindGroup({
      layout: this.pipeSunDepthBlock.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sceneUniformBuffer } },
        { binding: 1, resource: { buffer: this.blockStorageBuffer } },
      ],
    });

    this.bgSunDepthShard = d.createBindGroup({
      layout: this.pipeSunDepthShard.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sceneUniformBuffer } },
        { binding: 2, resource: { buffer: this.shardStorageBuffer } },
        { binding: 5, resource: { buffer: this.shardVertexBuffer } },
      ],
    });

    this.bgOpticalDepthBlock = d.createBindGroup({
      layout: this.pipeOpticalDepthBlock.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sceneUniformBuffer } },
        { binding: 1, resource: { buffer: this.blockStorageBuffer } },
      ],
    });

    this.bgOpticalDepthShard = d.createBindGroup({
      layout: this.pipeOpticalDepthShard.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sceneUniformBuffer } },
        { binding: 2, resource: { buffer: this.shardStorageBuffer } },
        { binding: 5, resource: { buffer: this.shardVertexBuffer } },
      ],
    });

    this.bgBlockCaustic = d.createBindGroup({
      layout: this.pipeBlockCaustic.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sceneUniformBuffer } },
        { binding: 1, resource: { buffer: this.blockStorageBuffer } },
        { binding: 3, resource: this.sunOpticalDepthView },
        { binding: 4, resource: this.sunDepthView },
      ],
    });

    this.bgShardCaustic = d.createBindGroup({
      layout: this.pipeShardCaustic.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sceneUniformBuffer } },
        { binding: 2, resource: { buffer: this.shardStorageBuffer } },
        { binding: 5, resource: { buffer: this.shardVertexBuffer } },
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
        { binding: 4, resource: { buffer: this.shardStorageBuffer } },
        { binding: 5, resource: this.sunDepthView },
        { binding: 6, resource: this.sunOpticalDepthView },
      ],
    });

    this.bgTowerDepth = d.createBindGroup({
      layout: this.pipeTowerDepth.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sceneUniformBuffer } },
        { binding: 1, resource: { buffer: this.blockStorageBuffer } },
      ],
    });

    this.bgGlassTower = d.createBindGroup({
      layout: this.pipeGlassTower.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sceneUniformBuffer } },
        { binding: 1, resource: { buffer: this.blockStorageBuffer } },
        { binding: 2, resource: this.causticView },
        { binding: 3, resource: this.linearSampler },
        { binding: 4, resource: this.sunDepthView },
        { binding: 5, resource: { buffer: this.shardStorageBuffer } },
        { binding: 6, resource: this.sunOpticalDepthView },
        { binding: 7, resource: { buffer: this.blockCentersUniformBuffer } },
      ],
    });

    this.bgSplinters = d.createBindGroup({
      layout: this.pipeSplinters.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sceneUniformBuffer } },
        { binding: 4, resource: { buffer: this.splinterStorageBuffer } },
        { binding: 5, resource: this.sunDepthView },
        { binding: 6, resource: this.sunOpticalDepthView },
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
    const dpr = Math.min(window.devicePixelRatio || 1, 2.0);
    const w = Math.max(64, Math.floor(cssWidth * dpr));
    const h = Math.max(64, Math.floor(cssHeight * dpr));
    if (w === this.width && h === this.height) return;

    this.width = w;
    this.height = h;
    this.canvas.width = w;
    this.canvas.height = h;

    this.hdrResolveTex?.destroy();
    this.refractionSnapshotTex?.destroy();
    this.depthResolveTex?.destroy();
    this.bloomTexA?.destroy();
    this.bloomTexB?.destroy();
    this.bloomWideTexA?.destroy();
    this.bloomWideTexB?.destroy();

    const d = this.device;

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

    this.depthResolveTex = d.createTexture({
      size: [w, h, 1],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.depthResolveView = this.depthResolveTex.createView();

    const bw = Math.max(32, Math.floor(w * 0.5));
    const bh = Math.max(32, Math.floor(h * 0.5));
    this.bloomW = bw;
    this.bloomH = bh;

    const ww = Math.max(16, Math.floor(w / 6));
    const wh = Math.max(16, Math.floor(h / 6));

    const mkBloomTex = (tw: number, th: number) =>
      d.createTexture({
        size: [tw, th, 1],
        format: 'rgba16float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });

    this.bloomTexA = mkBloomTex(bw, bh);
    this.bloomViewA = this.bloomTexA.createView();
    this.bloomTexB = mkBloomTex(bw, bh);
    this.bloomViewB = this.bloomTexB.createView();

    this.bloomWideTexA = mkBloomTex(ww, wh);
    this.bloomWideViewA = this.bloomWideTexA.createView();
    this.bloomWideTexB = mkBloomTex(ww, wh);
    this.bloomWideViewB = this.bloomWideTexB.createView();

    // Update static post-process blur/downsample step sizes on resize
    d.queue.writeBuffer(this.postUniformExtract, 0, new Float32Array([1 / w, 1 / h, 0, 0, 0, 0, 0, 0]));
    d.queue.writeBuffer(this.postUniformBlurH, 0, new Float32Array([1 / bw, 0.0, 0, 0, 0, 0, 0, 0]));
    d.queue.writeBuffer(this.postUniformBlurV, 0, new Float32Array([0.0, 1 / bh, 0, 0, 0, 0, 0, 0]));
    d.queue.writeBuffer(this.postUniformDownsample, 0, new Float32Array([1 / bw, 1 / bh, 0, 0, 0, 0, 0, 0]));
    d.queue.writeBuffer(this.postUniformWideBlurH, 0, new Float32Array([1 / ww, 0.0, 0, 0, 0, 0, 0, 0]));
    d.queue.writeBuffer(this.postUniformWideBlurV, 0, new Float32Array([0.0, 1 / wh, 0, 0, 0, 0, 0, 0]));

    this.bgGlassShard = d.createBindGroup({
      layout: this.pipeGlassShard.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sceneUniformBuffer } },
        { binding: 1, resource: { buffer: this.shardStorageBuffer } },
        { binding: 2, resource: this.refractionSnapshotView },
        { binding: 3, resource: this.linearSampler },
        { binding: 5, resource: this.sunDepthView },
        { binding: 6, resource: this.sunOpticalDepthView },
        { binding: 7, resource: { buffer: this.shardVertexBuffer } },
      ],
    });

    this.bgBloomExtract = d.createBindGroup({
      layout: this.pipeBloomExtract.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.postUniformExtract } },
        { binding: 1, resource: this.hdrResolveView },
        { binding: 3, resource: this.linearSampler },
      ],
    });

    this.bgBloomBlurH = d.createBindGroup({
      layout: this.pipeBloomBlur.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.postUniformBlurH } },
        { binding: 1, resource: this.bloomViewA },
        { binding: 3, resource: this.linearSampler },
      ],
    });

    this.bgBloomBlurV = d.createBindGroup({
      layout: this.pipeBloomBlur.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.postUniformBlurV } },
        { binding: 1, resource: this.bloomViewB },
        { binding: 3, resource: this.linearSampler },
      ],
    });

    this.bgBloomDownsample = d.createBindGroup({
      layout: this.pipeBloomDownsample.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.postUniformDownsample } },
        { binding: 1, resource: this.bloomViewA },
        { binding: 3, resource: this.linearSampler },
      ],
    });

    this.bgBloomWideBlurH = d.createBindGroup({
      layout: this.pipeBloomBlur.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.postUniformWideBlurH } },
        { binding: 1, resource: this.bloomWideViewA },
        { binding: 3, resource: this.linearSampler },
      ],
    });

    this.bgBloomWideBlurV = d.createBindGroup({
      layout: this.pipeBloomBlur.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.postUniformWideBlurV } },
        { binding: 1, resource: this.bloomWideViewB },
        { binding: 3, resource: this.linearSampler },
      ],
    });

    this.bgComposite = d.createBindGroup({
      layout: this.bglComposite,
      entries: [
        { binding: 0, resource: { buffer: this.postUniformComposite } },
        { binding: 1, resource: this.hdrResolveView },
        { binding: 2, resource: this.bloomViewA },
        { binding: 3, resource: this.linearSampler },
        { binding: 4, resource: this.bloomWideViewA },
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

    const camX = camera.position[0];
    const camY = camera.position[1];
    const camZ = camera.position[2];

    // 2. Sort intact stained-glass blocks by Y ascending (Early-Z is pre-populated by pipeTowerDepth!)
    // so uBlockCenters[0..intactCount-1] supports 7-step uniform binary search + early Y break.
    let intactCount = 0;
    let blkMinX = 1e6;
    let blkMinY = 1e6;
    let blkMinZ = 1e6;
    let blkMaxX = -1e6;
    let blkMaxY = -1e6;
    let blkMaxZ = -1e6;
    let maxBlockHalfY = 0.080;

    if (!this.benchHideBlocks) {
      for (let i = 0; i < physics.blocks.length && intactCount < MAX_BLOCKS_GPU; i++) {
        const b = physics.blocks[i];
        if (b.shattered) continue;
        this.blockSortDepths[i] = b.position[1];
        this.blockSortOrder[intactCount] = i;
        intactCount++;
      }

      for (let i = 1; i < intactCount; i++) {
        const keyIdx = this.blockSortOrder[i];
        const keyY = this.blockSortDepths[keyIdx];
        let j = i - 1;
        while (j >= 0 && this.blockSortDepths[this.blockSortOrder[j]] > keyY) {
          this.blockSortOrder[j + 1] = this.blockSortOrder[j];
          j--;
        }
        this.blockSortOrder[j + 1] = keyIdx;
      }

      for (let k = 0; k < intactCount; k++) {
        const b = physics.blocks[this.blockSortOrder[k]];
        const base = k * BLOCK_FLOATS;
        mat4FromRotationTranslationScale(this.tmpMatA, b.rotation, b.position, [1, 1, 1]);
        mat4FromRigidInverse(this.tmpMatB, b.rotation, b.position);

        this.blockStorageData.set(this.tmpMatA, base + 0);
        this.blockStorageData.set(this.tmpMatB, base + 16);

        const swatch = STAINED_GLASS_PALETTE[b.swatchId];
        const worldHalfX =
          Math.abs(this.tmpMatA[0]) * 0.375 +
          Math.abs(this.tmpMatA[4]) * 0.075 +
          Math.abs(this.tmpMatA[8]) * 0.125 +
          0.025;
        const worldHalfY =
          Math.abs(this.tmpMatA[1]) * 0.375 +
          Math.abs(this.tmpMatA[5]) * 0.075 +
          Math.abs(this.tmpMatA[9]) * 0.125 +
          0.005;
        const worldHalfZ =
          Math.abs(this.tmpMatA[2]) * 0.375 +
          Math.abs(this.tmpMatA[6]) * 0.075 +
          Math.abs(this.tmpMatA[10]) * 0.125 +
          0.025;

        const bx = b.position[0];
        const by = b.position[1];
        const bz = b.position[2];
        if (worldHalfY > maxBlockHalfY) maxBlockHalfY = worldHalfY;
        if (bx - worldHalfX < blkMinX) blkMinX = bx - worldHalfX;
        if (by - worldHalfY - 0.025 < blkMinY) blkMinY = by - worldHalfY - 0.025;
        if (bz - worldHalfZ < blkMinZ) blkMinZ = bz - worldHalfZ;
        if (bx + worldHalfX > blkMaxX) blkMaxX = bx + worldHalfX;
        if (by + worldHalfY + 0.025 > blkMaxY) blkMaxY = by + worldHalfY + 0.025;
        if (bz + worldHalfZ > blkMaxZ) blkMaxZ = bz + worldHalfZ;

        const cBase = k * 4;
        this.blockCentersData[cBase + 0] = bx;
        this.blockCentersData[cBase + 1] = by;
        this.blockCentersData[cBase + 2] = bz;
        this.blockCentersData[cBase + 3] = worldHalfY;

        this.blockStorageData[base + 32] = swatch.absorption[0];
        this.blockStorageData[base + 33] = swatch.absorption[1];
        this.blockStorageData[base + 34] = swatch.absorption[2];
        this.blockStorageData[base + 35] = worldHalfY;

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
        this.blockStorageData[base + 47] =
          isGrab > 0.5 ? physics.grabResistance01 : physics.getBlockPhotoelasticStress01(b.id);
        physics.writeBlockPhotoelasticWrench(b.id, this.blockStorageData, base + 48);
        if (isGrab > 0.5 && physics.grabResistance01 > 0.04) {
          const grabShear = physics.grabResistance01 * 0.62;
          this.blockStorageData[base + 51] = Math.max(this.blockStorageData[base + 51], grabShear);
        }
      }
    }

    if (intactCount > 0) {
      this.blockCentersData[96 * 4 + 0] = blkMinX;
      this.blockCentersData[96 * 4 + 1] = blkMinY;
      this.blockCentersData[96 * 4 + 2] = blkMinZ;
      this.blockCentersData[96 * 4 + 3] = maxBlockHalfY;
      this.blockCentersData[97 * 4 + 0] = blkMaxX;
      this.blockCentersData[97 * 4 + 1] = blkMaxY;
      this.blockCentersData[97 * 4 + 2] = blkMaxZ;
      this.blockCentersData[97 * 4 + 3] = this.polariscopeMode ? 1.0 : 0.0;

      this.device.queue.writeBuffer(
        this.blockCentersUniformBuffer,
        0,
        this.blockCentersData.buffer,
        0,
        98 * 4 * 4
      );
      this.device.queue.writeBuffer(
        this.blockStorageBuffer,
        0,
        this.blockStorageData.buffer,
        0,
        intactCount * BLOCK_FLOATS * 4
      );
    }

    // 3. Sort active 3D Voronoi glass shards back-to-front FIRST, then pack shardStorageData in sorted order
    // so uShards[k] matches sorted draw order across all shaders!
    const shardCount = this.benchHideBlocks ? 0 : Math.min(physics.shards.length, MAX_SHARDS_GPU);

    for (let i = 0; i < shardCount; i++) {
      const s = physics.shards[i];
      const dx = s.position[0] - camX;
      const dy = s.position[1] - camY;
      const dz = s.position[2] - camZ;
      this.shardSortDepths[i] = dx * dx + dy * dy + dz * dz;
      this.shardSortOrder[i] = i;
    }

    // In-place insertion sort farthest-to-nearest (k = 0 farthest, k = shardCount - 1 nearest)
    for (let i = 1; i < shardCount; i++) {
      const keyIdx = this.shardSortOrder[i];
      const keyDepth = this.shardSortDepths[keyIdx];
      let j = i - 1;
      while (j >= 0 && this.shardSortDepths[this.shardSortOrder[j]] < keyDepth) {
        this.shardSortOrder[j + 1] = this.shardSortOrder[j];
        j--;
      }
      this.shardSortOrder[j + 1] = keyIdx;
    }

    for (let k = 0; k < shardCount; k++) {
      const s = physics.shards[this.shardSortOrder[k]];
      const base = k * SHARD_FLOATS;
      mat4FromRotationTranslationScale(this.tmpMatA, s.rotation, s.position, [1, 1, 1]);
      this.shardStorageData.set(this.tmpMatA, base + 0);

      const swatch = STAINED_GLASS_PALETTE[s.swatchId];
      this.shardStorageData[base + 16] = swatch.absorption[0];
      this.shardStorageData[base + 17] = swatch.absorption[1];
      this.shardStorageData[base + 18] = swatch.absorption[2];
      // Pack (firstVertex << 8) | (vertexCount & 0xff) into native u32 GPUShardInstance.packedMesh (offset 19)
      this.shardStorageU32[base + 19] = ((s.firstVertex >>> 0) << 8) | (s.vertexCount & 0xff);

      this.shardStorageData[base + 20] = swatch.accentAbsorption[0];
      this.shardStorageData[base + 21] = swatch.accentAbsorption[1];
      this.shardStorageData[base + 22] = swatch.accentAbsorption[2];
      this.shardStorageData[base + 23] = s.rippleSeed;

      this.shardStorageData[base + 24] = s.impactLocalPos[0];
      this.shardStorageData[base + 25] = s.impactLocalPos[1];
      this.shardStorageData[base + 26] = s.impactLocalPos[2];
      this.shardStorageData[base + 27] = s.planeCount;

      this.shardStorageData.set(s.planes, base + 28);
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

    // 4. Pack grounded glass cullet chips
    let particleCount = 0;
    if (!this.benchHideBlocks) {
      for (let i = 0; i < physics.splinters.length && particleCount < MAX_SPLINTERS_GPU; i++) {
        const sp = physics.splinters[i];
        const base = particleCount * SPLINTER_FLOATS;
        const fade = 1.0 - Math.pow(sp.life / sp.maxLife, 3);
        this.splinterStorageData[base + 0] = sp.pos[0];
        this.splinterStorageData[base + 1] = sp.pos[1];
        this.splinterStorageData[base + 2] = sp.pos[2];
        this.splinterStorageData[base + 3] = sp.scale * fade;

        const sw = STAINED_GLASS_PALETTE[sp.swatchId];
        this.splinterStorageData[base + 4] = Math.exp(-sw.absorption[0] * 0.11);
        this.splinterStorageData[base + 5] = Math.exp(-sw.absorption[1] * 0.11);
        this.splinterStorageData[base + 6] = Math.exp(-sw.absorption[2] * 0.11);
        this.splinterStorageData[base + 7] = sp.rot;
        particleCount++;
      }
    }

    if (particleCount > 0) {
      this.device.queue.writeBuffer(
        this.splinterStorageBuffer,
        0,
        this.splinterStorageData.buffer,
        0,
        particleCount * SPLINTER_FLOATS * 4
      );
    }

    // 5. Pack spatial overlays (Free-Hand Top Surface Boundary + Live Landing Imprint & Spring Pull Filament)
    let slotDrawCount = 0;
    if (!this.benchHideBlocks && physics.turnPhase === 'EXTRACTED') {
      const carriedBlk = physics.activeBlockId !== null ? physics.blocks[physics.activeBlockId] : null;
      const sw = carriedBlk ? STAINED_GLASS_PALETTE[carriedBlk.swatchId] : STAINED_GLASS_PALETTE[2];
      const tintR = Math.max(0.45, Math.exp(-sw.absorption[0] * 0.14));
      const tintG = Math.max(0.45, Math.exp(-sw.absorption[1] * 0.14));
      const tintB = Math.max(0.45, Math.exp(-sw.absorption[2] * 0.14));

      const frame = physics.topSupportFrame;
      const padPos: [number, number, number] = [
        frame.surfaceCenter[0] + frame.upNormal[0] * 0.012,
        frame.surfaceCenter[1] + frame.upNormal[1] * 0.012,
        frame.surfaceCenter[2] + frame.upNormal[2] * 0.012,
      ];

      // Guide 0: Subtle 7.5 x 7.5 cm perimeter framing the open active top support surface
      mat4FromRotationTranslationScale(this.tmpMatA, frame.defaultRotation, padPos, [1.0, 0.14, 3.0]);
      this.overlaySlotData.set(this.tmpMatA, 0);
      this.overlaySlotData[16] = tintR;
      this.overlaySlotData[17] = tintG;
      this.overlaySlotData[18] = tintB;
      this.overlaySlotData[19] = 0.0;
      slotDrawCount = 1;

      // Guide 1: Crisp continuous landing footprint directly on the top row beneath the hovered block
      const imprintPos: [number, number, number] = [
        physics.carriedLandingPos[0] - frame.upNormal[0] * 0.056,
        physics.carriedLandingPos[1] - frame.upNormal[1] * 0.056,
        physics.carriedLandingPos[2] - frame.upNormal[2] * 0.056,
      ];
      mat4FromRotationTranslationScale(
        this.tmpMatA,
        physics.carriedTargetQuat,
        imprintPos,
        [0.996, 0.18, 0.992]
      );
      this.overlaySlotData.set(this.tmpMatA, 20);
      this.overlaySlotData[36] = tintR;
      this.overlaySlotData[37] = tintG;
      this.overlaySlotData[38] = tintB;
      this.overlaySlotData[39] = 1.0;
      slotDrawCount = 2;

      this.device.queue.writeBuffer(
        this.overlaySlotBuffer,
        0,
        this.overlaySlotData.buffer,
        0,
        slotDrawCount * 20 * 4
      );
    }

    const showSpring =
      !this.benchHideBlocks &&
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

    // 6. Compute Sun Orthographic View-Projection (exact 14.0 dm depth span [3.0, 17.0] for sub-mm PCSS!)
    const sunAzimuthDeg = this.lightingOverride?.sunAzimuthDeg ?? this.lightingPreset.sunAzimuthDeg;
    const sunElevationDeg =
      this.lightingOverride?.sunElevationDeg ?? this.lightingPreset.sunElevationDeg;
    const sunColor = this.lightingOverride?.sunColor ?? this.lightingPreset.sunColor;
    const skyTint = this.lightingOverride?.skyTint ?? this.lightingPreset.skyTint;

    const az = (sunAzimuthDeg * Math.PI) / 180;
    const el = (sunElevationDeg * Math.PI) / 180;
    const sunDir: Vec3 = [
      Math.cos(el) * Math.sin(az),
      Math.sin(el),
      Math.cos(el) * Math.cos(az),
    ];

    const towerTopDm = intactCount > 0 ? Math.max(0.0, blkMaxY) : 0.0;
    const sunTargetY =
      towerTopDm > 2.82 ? 1.25 + (towerTopDm - 2.82) * 0.45 : 1.25;
    const sunEye: Vec3 = [sunDir[0] * 10.0, sunTargetY + sunDir[1] * 10.0, sunDir[2] * 10.0];
    mat4LookAt(this.sunViewMatrix, sunEye, [0, sunTargetY, 0], [0, 1, 0]);
    mat4OrthoZO(this.sunProjMatrix, -4.85, 4.85, -4.85, 4.85, 3.0, 17.0);
    mat4Multiply(this.sunViewProjMatrix, this.sunProjMatrix, this.sunViewMatrix);

    const aspect = this.width / Math.max(1, this.height);
    this.sceneUniformData.set(camera.viewProjMatrix, 0);
    this.sceneUniformData.set(camera.invViewProjMatrix, 16);
    this.sceneUniformData.set(this.sunViewProjMatrix, 32);

    this.sceneUniformData[48] = camX;
    this.sceneUniformData[49] = camY;
    this.sceneUniformData[50] = camZ;
    this.sceneUniformData[51] =
      this.referenceCardLuminance > 0 ? -this.referenceCardLuminance : elapsedSec;

    this.sceneUniformData[52] = sunDir[0];
    this.sceneUniformData[53] = sunDir[1];
    this.sceneUniformData[54] = sunDir[2];
    this.sceneUniformData[55] = intactCount;

    this.sceneUniformData[56] = sunColor[0];
    this.sceneUniformData[57] = sunColor[1];
    this.sceneUniformData[58] = sunColor[2];
    this.sceneUniformData[59] = aspect;

    this.sceneUniformData[60] = skyTint[0];
    this.sceneUniformData[61] = skyTint[1];
    this.sceneUniformData[62] = skyTint[2];
    this.sceneUniformData[63] = shardCount;

    let aoMinX = 1e6;
    let aoMinZ = 1e6;
    let aoMaxX = -1e6;
    let aoMaxZ = -1e6;

    for (let k = 0; k < intactCount; k++) {
      const b = physics.blocks[this.blockSortOrder[k]];
      const by = b.position[1];
      if (by >= 0.52) break;
      const bx = b.position[0];
      const bz = b.position[2];
      if (bx - 0.50 < aoMinX) aoMinX = bx - 0.50;
      if (bz - 0.50 < aoMinZ) aoMinZ = bz - 0.50;
      if (bx + 0.50 > aoMaxX) aoMaxX = bx + 0.50;
      if (bz + 0.50 > aoMaxZ) aoMaxZ = bz + 0.50;
    }

    const shardStride = shardCount > 36 ? Math.ceil(shardCount / 36) : 1;
    for (let k = 0; k < shardCount; k += shardStride) {
      const s = physics.shards[this.shardSortOrder[k]];
      const sy = s.position[1];
      if (sy < 0.28) {
        const sx = s.position[0];
        const sz = s.position[2];
        const eqR = Math.min(0.14, Math.max(0.018, Math.cbrt(Math.max(1e-6, s.volume) * 0.2387324)));
        const pad = eqR + 0.08;
        if (sx - pad < aoMinX) aoMinX = sx - pad;
        if (sz - pad < aoMinZ) aoMinZ = sz - pad;
        if (sx + pad > aoMaxX) aoMaxX = sx + pad;
        if (sz + pad > aoMaxZ) aoMaxZ = sz + pad;
      }
    }

    if (aoMinX > aoMaxX) {
      this.sceneUniformData[64] = 10.0;
      this.sceneUniformData[65] = 10.0;
      this.sceneUniformData[66] = -10.0;
      this.sceneUniformData[67] = -10.0;
    } else {
      this.sceneUniformData[64] = aoMinX;
      this.sceneUniformData[65] = aoMinZ;
      this.sceneUniformData[66] = aoMaxX;
      this.sceneUniformData[67] = aoMaxZ;
    }
    this.sceneUniformData[68] = intactCount;
    this.sceneUniformData[69] = shardStride;
    this.sceneUniformData[70] = maxBlockHalfY;
    this.sceneUniformData[71] = this.lightingPreset.atelierMix ?? 0;

    this.device.queue.writeBuffer(this.sceneUniformBuffer, 0, this.sceneUniformData.buffer);

    // Update camera-coupled macro Depth-of-Field & optical composite parameters
    const focusDistDm = Math.hypot(
      camX - camera.target[0],
      camY - camera.target[1],
      camZ - camera.target[2]
    );
    this.compositeUniformData[0] = 1 / this.bloomW;
    this.compositeUniformData[1] = 1 / this.bloomH;
    this.compositeUniformData[2] = 0.085; // Tight physical optical glare
    this.compositeUniformData[3] = 1.18;  // Calibrated exposure for low-key dark walnut stage
    this.compositeUniformData[4] = focusDistDm;
    this.compositeUniformData[5] = Math.max(1.65, towerTopDm * 0.58); // +/- 16.5 cm tack-sharp zone (expands automatically on tall >18-row stacks)
    this.compositeUniformData[6] = 4.2;   // Subtle natural f/2.8 bokeh blur on far table rim & backdrop
    this.compositeUniformData[7] = aspect;
    this.device.queue.writeBuffer(this.postUniformComposite, 0, this.compositeUniformData.buffer);

    // 7. Record all GPU Passes into ONE single CommandEncoder!
    const swapchainView = this.context.getCurrentTexture().createView();
    const encoder = this.device.createCommandEncoder({ label: 'VitrailFrameEncoder' });

    // Pass 1a: 2048x2048 Sun Shadow Depth Map (first-surface depth for upstream glass identification)
    {
      const pass = encoder.beginRenderPass({
        label: 'SunShadowDepthPass',
        colorAttachments: [],
        depthStencilAttachment: {
          view: this.sunDepthView,
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
        timestampWrites: this.tsWritesStart,
      });

      if (intactCount > 0) {
        pass.setPipeline(this.pipeSunDepthBlock);
        pass.setBindGroup(0, this.bgSunDepthBlock);
        pass.setVertexBuffer(0, this.boxVertexBuffer);
        pass.setIndexBuffer(this.boxIndexBuffer, 'uint16');
        pass.drawIndexed(this.boxIndexCount, intactCount, 0, 0, 0);
      }

      if (shardCount > 0) {
        pass.setPipeline(this.pipeSunDepthShard);
        pass.setBindGroup(0, this.bgSunDepthShard);
        pass.draw(132, shardCount, 0, 0);
      }
      pass.end();
    }

    // Pass 1b: 1024x1024 Spectral Optical Depth (.rgb add) + Min-Caster-Height (.a min) Atlas
    {
      const pass = encoder.beginRenderPass({
        label: 'SunOpticalDepthPass',
        colorAttachments: [
          {
            view: this.sunOpticalDepthView,
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 10.0 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });

      if (intactCount > 0) {
        pass.setPipeline(this.pipeOpticalDepthBlock);
        pass.setBindGroup(0, this.bgOpticalDepthBlock);
        pass.setVertexBuffer(0, this.boxVertexBuffer);
        pass.setIndexBuffer(this.boxIndexBuffer, 'uint16');
        pass.drawIndexed(this.boxIndexCount, intactCount, 0, 0, 0);
      }

      if (shardCount > 0) {
        pass.setPipeline(this.pipeOpticalDepthShard);
        pass.setBindGroup(0, this.bgOpticalDepthShard);
        pass.draw(132, shardCount, 0, 0);
      }
      pass.end();
    }

    // Pass 1c: 1024x1024 Additive Snell-Refracted Jacobian Caustic Flux Map (src + dst)
    {
      const pass = encoder.beginRenderPass({
        label: 'CausticFluxPass',
        colorAttachments: [
          {
            view: this.causticView,
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });

      if (intactCount > 0) {
        pass.setPipeline(this.pipeBlockCaustic);
        pass.setBindGroup(0, this.bgBlockCaustic);
        pass.setVertexBuffer(0, this.causticRefractVertexBuffer);
        pass.setIndexBuffer(this.causticRefractIndexBuffer, 'uint16');
        pass.drawIndexed(this.causticRefractIndexCount, intactCount, 0, 0, 0);
      }

      if (shardCount > 0) {
        pass.setPipeline(this.pipeShardCaustic);
        pass.setBindGroup(0, this.bgShardCaustic);
        pass.draw(132, shardCount, 0, 0);
      }
      pass.end();
    }

    // Pass 2 & 3: Studio Alcove + Bullnose Dark Walnut Table + Multi-Block Raytraced Glass Tower (1x Retina HDR)
    const hasShards = shardCount > 0;
    {
      const pass = encoder.beginRenderPass({
        label: 'MainOpaqueAndTowerPass',
        colorAttachments: [
          {
            view: this.hdrResolveView,
            clearValue: { r: 0.02, g: 0.02, b: 0.024, a: 14.0 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
        depthStencilAttachment: {
          view: this.depthResolveView,
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: hasShards ? 'store' : 'discard',
        },
      });

      if (intactCount > 0) {
        pass.setVertexBuffer(0, this.beveledVertexBuffer);
        pass.setIndexBuffer(this.beveledIndexBuffer, 'uint16');

        pass.setPipeline(this.pipeTowerDepth);
        pass.setBindGroup(0, this.bgTowerDepth);
        pass.drawIndexed(this.beveledIndexCount, intactCount, 0, 0, 0);

        pass.setPipeline(this.pipeGlassTower);
        pass.setBindGroup(0, this.bgGlassTower);
        pass.drawIndexed(this.beveledIndexCount, intactCount, 0, 0, 0);
      }

      pass.setPipeline(this.pipeTable);
      pass.setBindGroup(0, this.bgTable);
      pass.draw(192 * 33, 1, 0, 0);

      pass.setPipeline(this.pipeSky);
      pass.setBindGroup(0, this.bgSky);
      pass.draw(3, 1, 0, 0);

      if (!hasShards) {
        this.drawOverlays(pass, particleCount, slotDrawCount, showSpring);
      }

      pass.end();
    }

    if (hasShards) {
      // Copy resolved HDR scene (Table + Intact Tower) for Glass Shard refraction
      encoder.copyTextureToTexture(
        { texture: this.hdrResolveTex },
        { texture: this.refractionSnapshotTex },
        [this.width, this.height, 1]
      );

      // Multi-slice sorted 1x shard depth peeling with single-draw instanced storage pull:
      // 3 depth slices when shardCount > 24 so overlapping shard piles refract shards beneath them!
      const useThreeSlices = shardCount > 24;
      const split1 = useThreeSlices
        ? Math.floor(shardCount * 0.35)
        : shardCount > 1
        ? Math.floor(shardCount * 0.55)
        : 0;
      const split2 = useThreeSlices ? Math.floor(shardCount * 0.70) : split1;

      const drawShardRange = (label: string, startIdx: number, endIdx: number) => {
        if (endIdx <= startIdx) return;
        const slicePass = encoder.beginRenderPass({
          label,
          colorAttachments: [
            {
              view: this.hdrResolveView,
              loadOp: 'load',
              storeOp: 'store',
            },
          ],
          depthStencilAttachment: {
            view: this.depthResolveView,
            depthLoadOp: 'load',
            depthStoreOp: 'store',
          },
        });
        slicePass.setPipeline(this.pipeGlassShard);
        slicePass.setBindGroup(0, this.bgGlassShard);
        slicePass.draw(132, endIdx - startIdx, 0, startIdx);
        slicePass.end();

        encoder.copyTextureToTexture(
          { texture: this.hdrResolveTex },
          { texture: this.refractionSnapshotTex },
          [this.width, this.height, 1]
        );
      };

      drawShardRange('ShardsSlice1Pass', 0, split1);
      drawShardRange('ShardsSlice2Pass', split1, split2);

      // Final foreground Voronoi Glass Shards + Micro-Splinters + Spatial Placement Guides (1x direct HDR resolve)
      {
        const pass = encoder.beginRenderPass({
          label: 'ShardsFrontAndOverlaysPass',
          colorAttachments: [
            {
              view: this.hdrResolveView,
              loadOp: 'load',
              storeOp: 'store',
            },
          ],
          depthStencilAttachment: {
            view: this.depthResolveView,
            depthLoadOp: 'load',
            depthStoreOp: 'discard',
          },
        });

        if (shardCount > split2) {
          pass.setPipeline(this.pipeGlassShard);
          pass.setBindGroup(0, this.bgGlassShard);
          pass.draw(132, shardCount - split2, 0, split2);
        }

        this.drawOverlays(pass, particleCount, slotDrawCount, showSpring);
        pass.end();
      }
    }

    // Pass 5a: Karis Anti-Firefly HDR Bloom Extract (1/2 res)
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

    // Pass 5b: Tight Glare Blur (1/2 res H -> V) + Wide Halation Downsample & Blur (1/6 res H -> V)
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

      const passDown = encoder.beginRenderPass({
        label: 'BloomWideDownsamplePass',
        colorAttachments: [
          { view: this.bloomWideViewA, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } },
        ],
      });
      passDown.setPipeline(this.pipeBloomDownsample);
      passDown.setBindGroup(0, this.bgBloomDownsample);
      passDown.draw(3, 1, 0, 0);
      passDown.end();

      const passWideH = encoder.beginRenderPass({
        label: 'BloomWideBlurHPass',
        colorAttachments: [
          { view: this.bloomWideViewB, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } },
        ],
      });
      passWideH.setPipeline(this.pipeBloomBlur);
      passWideH.setBindGroup(0, this.bgBloomWideBlurH);
      passWideH.draw(3, 1, 0, 0);
      passWideH.end();

      const passWideV = encoder.beginRenderPass({
        label: 'BloomWideBlurVPass',
        colorAttachments: [
          { view: this.bloomWideViewA, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } },
        ],
      });
      passWideV.setPipeline(this.pipeBloomBlur);
      passWideV.setBindGroup(0, this.bgBloomWideBlurV);
      passWideV.draw(3, 1, 0, 0);
      passWideV.end();
    }

    // Pass 5c: Depth-of-Field + Dual-Mip Optical Bloom + Filmic Tone Map -> Swapchain
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
        timestampWrites: this.tsWritesEnd,
      });
      pass.setPipeline(this.pipeComposite);
      pass.setBindGroup(0, this.bgComposite);
      pass.draw(3, 1, 0, 0);
      pass.end();
    }

    const canReadTimestamp =
      this.supportsTimestamps &&
      this.timestampQuerySet !== null &&
      this.timestampResolveBuffer !== null &&
      this.timestampReadBuffer !== null &&
      !this.timestampReadPending;

    if (canReadTimestamp) {
      encoder.resolveQuerySet(this.timestampQuerySet!, 0, 2, this.timestampResolveBuffer!, 0);
      encoder.copyBufferToBuffer(this.timestampResolveBuffer!, 0, this.timestampReadBuffer!, 0, 16);
    }

    this.device.queue.submit([encoder.finish()]);

    if (canReadTimestamp) {
      this.timestampReadPending = true;
      const readBuf = this.timestampReadBuffer!;
      readBuf
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          if (this.destroyed) return;
          const times = new BigInt64Array(readBuf.getMappedRange());
          const deltaNs = times[1] > times[0] ? Number(times[1] - times[0]) : 0;
          readBuf.unmap();
          if (deltaNs > 0 && deltaNs < 500_000_000) {
            this.lastGpuFrameMs = deltaNs * 1e-6;
          }
          this.timestampReadPending = false;
        })
        .catch(() => {
          this.timestampReadPending = false;
        });
    }
  }

  /**
   * Reads back the linear `rgba16float` HDR resolve target (`hdrResolveTex`) as IEEE-754 `Float32Array`
   * and counts any `NaN` or `Inf` channels for Tier 2 optical bench radiometry.
   */
  public async readHdrResolveFloat32(): Promise<{
    width: number;
    height: number;
    nanInfCount: number;
    rgba: Float32Array;
  }> {
    const w = this.width;
    const h = this.height;
    const bytesPerRow = Math.ceil((w * 8) / 256) * 256;
    const byteSize = bytesPerRow * h;

    const readBuf = this.device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const enc = this.device.createCommandEncoder({ label: 'HdrReadbackEncoder' });
    enc.copyTextureToBuffer(
      { texture: this.hdrResolveTex },
      { buffer: readBuf, bytesPerRow, rowsPerImage: h },
      [w, h, 1]
    );
    this.device.queue.submit([enc.finish()]);

    await readBuf.mapAsync(GPUMapMode.READ);
    const mapped = new Uint16Array(readBuf.getMappedRange());
    const u16PerRow = bytesPerRow >>> 1;
    const rgba = new Float32Array(w * h * 4);
    let nanInfCount = 0;

    for (let y = 0; y < h; y++) {
      const rowSrc = y * u16PerRow;
      const rowDst = y * w * 4;
      for (let c = 0; c < w * 4; c++) {
        const bits = mapped[rowSrc + c];
        const sign = (bits & 0x8000) !== 0 ? -1 : 1;
        const exp = (bits >>> 10) & 0x1f;
        const frac = bits & 0x03ff;
        let val = 0;
        if (exp === 0) {
          val = frac === 0 ? 0 : sign * Math.pow(2, -14) * (frac / 1024);
        } else if (exp === 0x1f) {
          nanInfCount++;
          val = frac === 0 ? sign * Infinity : NaN;
        } else {
          val = sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
        }
        rgba[rowDst + c] = val;
      }
    }

    readBuf.unmap();
    readBuf.destroy();
    return { width: w, height: h, nanInfCount, rgba };
  }

  private drawOverlays(
    pass: GPURenderPassEncoder,
    particleCount: number,
    slotDrawCount: number,
    showSpring: boolean
  ): void {
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
  }

  public dispose(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    try {
      if (this.hdrMediaQuery && this.onHdrMediaChange) {
        this.hdrMediaQuery.removeEventListener('change', this.onHdrMediaChange);
        this.hdrMediaQuery = null;
        this.onHdrMediaChange = null;
      }
      this.context?.unconfigure();
      this.timestampQuerySet?.destroy();
      this.timestampResolveBuffer?.destroy();
      this.timestampReadBuffer?.destroy();
      this.sceneUniformBuffer?.destroy();
      this.blockCentersUniformBuffer?.destroy();
      this.blockStorageBuffer?.destroy();
      this.shardStorageBuffer?.destroy();
      this.splinterStorageBuffer?.destroy();
      this.overlaySlotBuffer?.destroy();
      this.springUniformBuffer?.destroy();
      this.postUniformExtract?.destroy();
      this.postUniformBlurH?.destroy();
      this.postUniformBlurV?.destroy();
      this.postUniformDownsample?.destroy();
      this.postUniformWideBlurH?.destroy();
      this.postUniformWideBlurV?.destroy();
      this.postUniformComposite?.destroy();
      this.beveledVertexBuffer?.destroy();
      this.beveledIndexBuffer?.destroy();
      this.causticRefractVertexBuffer?.destroy();
      this.causticRefractIndexBuffer?.destroy();
      this.boxVertexBuffer?.destroy();
      this.boxIndexBuffer?.destroy();
      this.shardVertexBuffer?.destroy();
      this.sunDepthTex?.destroy();
      this.sunOpticalDepthTex?.destroy();
      this.causticTex?.destroy();
      this.hdrResolveTex?.destroy();
      this.refractionSnapshotTex?.destroy();
      this.depthResolveTex?.destroy();
      this.bloomTexA?.destroy();
      this.bloomTexB?.destroy();
      this.bloomWideTexA?.destroy();
      this.bloomWideTexB?.destroy();
      this.device?.destroy();
    } catch {
      // Ignore cleanup errors on already-lost GPUDevice
    }
  }
}
