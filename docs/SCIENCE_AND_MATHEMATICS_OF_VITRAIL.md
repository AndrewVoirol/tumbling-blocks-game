# The Science and Mathematics of Vitrail

*Vitrail* replaces the opaque wooden blocks of regulation tumbling-block towers with solid cathedral stained-glass prisms resting on a circular quarter-sawn oak pedestal table. That single material substitution couples four domains of classical and computational physics that rarely interact inside one real-time browser loop:

1. **Multibody contact mechanics and dry glass tribology** (MuJoCo 4.0 sparse Newton solver, elliptic friction cones, Stribeck slip, and Housner rocking).
2. **Linear-elastic brittle fracture** (Griffith sub-critical crazing softening, three-point Euler-Bernoulli flexural snap, and momentum-conserving 3D anisotropic Voronoi cleavage).
3. **Physical optics and photoelasticity** (Snell refraction, Cauchy dispersion, exact dielectric Fresnel and Total Internal Reflection, Beer-Lambert spectral attenuation, Jacobian sun caustics, and dark-field Maxwell-Brewster stress birefringence).
4. **Continuum modal and impact acoustics** (Timoshenko thick-bar dispersion with solder-came damping, Hertzian contact-duration spectral filtering, orthotropic Kirchhoff plate Bessel modes, and elastodynamic shard scaling).

This essay documents the governing equations, numerical discretizations, and engineering trade-offs implemented in the codebase.

---

## 1. Why Stained Glass? Force Chains as Light, Failure as Geometry and Sound

In an opaque wooden tower, structural stability is hidden: a player discovers whether a block carries a critical column load only by poking it. In annealed soda-lime glass ($n_d = 1.522$), mechanical state maps directly onto optical and acoustic observables:

- **Internal load paths become visible.** Under cross-polarized illumination (Polariscope mode), compressive clamping forces and flexural bending moments retard orthogonal polarization states proportionally to principal stress difference (Brewster, 1815; Maxwell, 1853), revealing live isochromatic fringes along the load-bearing columns while unloaded loose blocks extinguish dark.
- **Sub-critical impacts leave structural memory.** Impacts above the crazing threshold nucleate internal radial cleavage planes and a Hertzian impact cone around the local strike coordinate, simultaneously scattering internal light rays, lowering subsequent Griffith fracture thresholds, and detuning modal vibration frequencies.
- **Catastrophic collapse is non-recoverable.** Over-stressed prisms cleave into 3D convex Voronoi shards that conserve linear and angular momentum, project colored refracted caustics onto the tabletop, extinguish in dark-field polariscopy save for frozen cleavage-edge strain, and ring at frequencies inversely proportional to the cube root of shard mass.

### Dimensional Scaling and Froude Similarity

Regulation tumbling-block prisms measure $L \times b \times h = 75 \times 25 \times 15\text{ mm}$ ($3:1:0.6$ aspect ratio) with $1.25\text{ mm}$ diamond-ground $45^\circ$ edge chamfers. At bulk leaded stained-glass effective density ($\rho_{\text{eff}} \approx 2520\text{ kg/m}^3$), each intact block has mass $m = 71.1\text{ g}$ ($0.0711\text{ kg}$), giving an 18-course, 54-block tower a total mass of $3.84\text{ kg}$ and stack height of $27.0\text{ cm}$.

Inside the MuJoCo 4.0 WASM solver (`src/physics/mujocoTowerBuilder.ts`), all mass, length, force, and inertia quantities are integrated in strict SI units ($\text{m}, \text{kg}, \text{s}, \text{N}$) under terrestrial gravity $g = 9.81\text{ m/s}^2$ (`1.000g`). Externally, camera and rendering coordinates operate in decimeters ($1\text{ unit} = 0.1\text{ m}$, so block half-extents are $0.375 \times 0.075 \times 0.125\text{ dm}$) linked by exact Froude similarity ($Fr = v / \sqrt{g \ell} = 1$):

$$
\ell_{\text{dm}} = 10\,\ell_{\text{SI}}, \qquad t_{\text{sim}} = \sqrt{10}\,t_{\text{SI}}, \qquad v_{\text{sim}} = \sqrt{10}\,v_{\text{SI}}, \qquad \boldsymbol{\omega}_{\text{sim}} = \frac{\boldsymbol{\omega}_{\text{SI}}}{\sqrt{10}}
$$

Under this transformation, gravitational acceleration in simulation units is $g_{\text{sim}} = 9.81\text{ dm/s}_{\text{sim}}^2$, mapping back to $g_{\text{real}} = (\sqrt{10})^2 \cdot 0.981 = 9.81\text{ m/s}^2$ without time-dilation drift (`SIM_TIME_SCALE` $= \sqrt{10} \approx 3.162278$ in `src/physics/constants.ts`).

---

## 2. Contact Mechanics, Tribology & Rocking Stability

### Convex Soft-Contact Formulation and Sparse Newton Solver

An 18-to-28-course glass tower is a statically indeterminate assembly of up to $84$ rigid bodies ($504$ unconstrained degrees of freedom) coupled by hundreds of simultaneous face-on-face contact patches. Sequential impulse or projected Gauss-Seidel (PGS) game-physics solvers exhibit vertical compliance drift ("stack sag") and jitter unless artificially stabilized by aggressive velocity damping.

*Vitrail* runs Google DeepMind's `@mujoco/mujoco` 4.0 runtime inside a dedicated ES Module Web Worker (`src/physics/mujocoWorker.ts`, dispatching 4 WASM pthreads via `mju_threadpool` when cross-origin isolated) at fixed timestep $\Delta t = 1/600\text{ s}$ ($1.667\text{ ms}$). Following Gauss's Principle of Least Constraint (Gauss, 1829) and Todorov, Erez, and Tassa (2012), unconstrained accelerations $\mathbf{a}_u = \mathbf{M}^{-1}(\boldsymbol{\tau} - \mathbf{c})$ are regularized in dual/primal space by minimizing a strictly convex objective over generalized accelerations $\mathbf{a} \in \mathbb{R}^{n_v}$:

$$
\min_{\mathbf{a}} \; \frac{1}{2}(\mathbf{a} - \mathbf{a}_u)^\top \mathbf{M} (\mathbf{a} - \mathbf{a}_u) + \sum_{i} s_i\!\left(\mathbf{J}_i \mathbf{a} - \mathbf{a}_{0,i}\right)
$$

where $\mathbf{M} \in \mathbb{R}^{n_v \times n_v}$ is the sparse joint-space inertia matrix, $\mathbf{J}_i$ is the contact Jacobian row block, $\mathbf{a}_{0,i}$ is the Kelvin-Voigt reference acceleration governed by `solref` $= (\tau_{\text{ref}}, \zeta)$ and impedance `solimp`, and $s_i(\cdot)$ penalizes constraint violation across elliptic friction cones. Using `jacobian="sparse"` and `solver="Newton"` (`26` max Newton iterations upright), Cholesky factorization of the exact Hessian converges to machine precision in $2\text{–}5$ iterations per substep: calibrated static stack sag across $18$ courses is $0.098\text{ mm}$ ($< 0.15\text{ mm}$ tolerance) and $0.199\text{ mm}$ across $24$ courses ($< 0.50\text{ mm}$ tolerance), while full-tower island sleep (`sleepTolerance = 0.003`) reduces idle solver cost to $8.2\text{ }\mu\text{s/frame}$.

### Elliptic Friction Cones ($\text{condim} \in \{4, 6\}$) and Stribeck Slip

Point contacts with pyramidal 3D Coulomb cones ($\text{condim} = 3$) exert zero torsional resistance about a single contact normal, allowing a narrow block pushed off-center to pivot with unphysical ease unless multiple contact vertices remain active. Every beveled glass block in `mujocoTowerBuilder.ts` uses `cone="elliptic"` with $\text{condim} = 4$ (normal force $f_n$, tangential shear $f_{t1}, f_{t2}$, and torsional spin torque $\tau_n$):

$$
\left(\frac{f_{t1}}{\mu_t}\right)^2 + \left(\frac{f_{t2}}{\mu_t}\right)^2 + \left(\frac{\tau_n}{\mu_{\text{spin}}}\right)^2 \le f_n^2
$$

with `glassFriction` $= (\mu_t, \mu_{\text{spin}}, \mu_{\text{roll}}) = (0.275, 0.0036, 0.00018)$ and `impratio` $= 4.5$. The circular oak tabletop uses $\text{condim} = 6$, adding two rolling-resistance axes ($\mu_{\text{roll}} = 0.00035$) so resting cylindrical/polyhedral fragments settle without perpetual micro-rolling.

Dry silicate glass exhibits pronounced velocity-weakening friction (Stribeck, 1902): adsorbed surface moisture and asperity junctions yield a static breakaway coefficient $\mu_s = 0.275$ that drops toward $\mu_k = 0.125$ once macroscopic sliding commences:

$$
\mu(v) = \mu_k + (\mu_s - \mu_k)\exp\!\left[-\left(\frac{v}{v_S}\right)^\delta\right]
$$

In `MujocoPhysicsEngine.ts`, blocks whose horizontal slip speed crosses $[0.014, 0.072]\text{ m/s}$ dynamically elevate `geom_priority[gid] = 1` with `glassKineticFriction` $= (0.125, 0.0016, 0.00012)$, reproducing the crisp breakaway-to-glide tactile transition during extraction while keeping all surrounding $\text{priority} = 0$ column contacts locked at $\mu_s = 0.275$.

### Isometric Grazing Drag-Plane Unfolding

Player extraction couples a kinematic `mocap` hand body to the clicked surface point via an implicit soft `<connect>` equality constraint (`handSolref = [0.024, 1.15]`), preventing infinite-force constraint fighting against clamped columns. When a player views a tall ($22\text{–}28$ course) stack from a low or upward camera elevation $\phi = \operatorname{atan2}(\Delta y_{\text{cam}}, d_{\text{horiz}}) < 25^\circ$, intersecting the mouse ray with a pure horizontal plane $y = y_0$ becomes ill-conditioned ($\mathbf{d}_{\text{ray}} \cdot \hat{\mathbf{y}} \to 0$).

To avoid both ray-horizon blowup and $\cos\beta$ radial foreshortening, `GameEngine.projectBlockDragPlane` tilts the virtual drag plane toward the camera by $\beta = \min(\pi/2, 25^\circ - \phi)$ with normal $\hat{\mathbf{n}} = (\hat{h}_x \sin\beta,\, \cos\beta,\, \hat{h}_z \sin\beta)$ and then **isometrically unfolds** the tilted displacement $\Delta\mathbf{r} = \mathbf{p}_{\text{hit}} - \mathbf{p}_{\text{anchor}}$ back onto $y = y_0$ around the lateral axis:

$$
d_{\text{lat}} = -\Delta r_x \hat{h}_z + \Delta r_z \hat{h}_x, \qquad d_{\text{rad}} = (\Delta r_x \hat{h}_x + \Delta r_z \hat{h}_z)\cos\beta - \Delta r_y \sin\beta
$$

This preserves $C^1$ continuity across $\phi = 25^\circ$, exact $0.000\text{ mm}$ grab-start jump, and $1.000\times$ radial pull gain even at $-65^\circ$ upward pitches.

### Housner Inverted-Pendulum Bevel Rocking

Each block is a 24-vertex convex hull with $1.25\text{ mm}$ chamfers, placing its transverse rocking pivots at $b_{\text{pivot}} = \frac{b}{2} - c_b = 11.25\text{ mm}$ from the centerline and $h/2 = 7.5\text{ mm}$ below the center of mass ($R_{\text{rock}} = \sqrt{b_{\text{pivot}}^2 + (h/2)^2} = 13.52\text{ mm}$). Following Housner (1963), the linearized equation of rocking about either pivot for $|\theta| < \alpha = \arctan(b_{\text{pivot}} / (h/2))$ is:

$$
I_O \ddot{\theta} = -m g R_{\text{rock}} \sin(\alpha - |\theta|)\operatorname{sgn}(\theta)
$$

where $I_O = I_{\text{cm}} + m R_{\text{rock}}^2$ is the moment of inertia about the pivot edge. In classical rigid-body Housner theory, conservation of angular momentum across an impulsive flat-face slap predicts angular velocity ratio $\dot{\theta}^+ / \dot{\theta}^- = 1 - \frac{3}{2}\sin^2\alpha_{\text{slenderness}}$. Because MuJoCo's primary block contact uses critical/overdamped Kelvin-Voigt reference `contactSolref = [0.0042, 1.12]` ($\zeta = 1.12 > 1$) to eliminate static stack bounce, a raw flat-face impact quenches $\dot{\theta}^+$ on the first slap. For unclamped prisms ($F_{\text{clamp}} < 1.10\text{ N}$) contacting at most two bodies after $\ge 3$ free-flight substeps, `MujocoPhysicsEngine.ts` restores dissipative soda-lime bevel pivot transfer $\omega_{k+1}^+ = e_{\text{rock},k}\,\omega_k^-$ with $e_{\text{rock},k} = e_0 \cdot 0.88^k$ ($e_0 \in \{0.74, 0.76\}$) and kinematic center-of-mass lift $v_y^+ = \omega_{k+1}^+ b_{\text{pivot}}$. This produces $3$ damped sign reversals on an $8^\circ$ ajar prism and $9$ accelerating reversals ($36.6\text{ ms} \to 10.0\text{ ms}$) on a $6.5^\circ$ rubble-fulcrum seesaw while emitting zero false wobble impulses on clamped lower courses.

### Native `mj_ray` Picking and Constraint-Island Sleep Hygiene

Interactive tap-scouting, drag anchoring, and Sandbox shard tap-to-flick (`nudgeShard`) query MuJoCo's native WASM raycaster `mj_ray` directly against `geom_group` $0\text{..}3$ ($0$ = intact beveled blocks, $1$ = convex Voronoi shards, $2$ = oak tabletop, $3$ = hidden parking shelf), achieving sub-millimeter intersection accuracy against the exact 24-vertex beveled prism hulls on both desktop pointers and iPad Pro multi-touch viewports. When a block is grabbed, nudged, or shattered, `collectConstraintIslandTrees` inspects MuJoCo 4.0's native constraint-island graph (`d.nisland`, `d.tree_island`, `d.tree_asleep`) to wake only topologically coupled trees rather than perturbing unrelated sleeping rubble.

Shattered intact-block slots are parked on a recessed `parking_shelf` geom (`contype = 0` on parked bodies, `priority = 10`, `condim = 3`, `friction = "1.0 0.02 0.01"`, `conaffinity = 0`). Giving the shelf finite frictional resistance at exact resting height guarantees that deactivated parent bodies enter native island sleep within milliseconds rather than sliding perpetually underground (`ntree_awake > 0`) and inadvertently holding the heavy-cascade `mjDSBL_MULTICCD` bypass open after a collapse settles. Across 8 seeded 18-course collapses, $100\%$ of intact survivors and Voronoi shards reach native sleep (`ntree_awake = 0`) within $1.12\text{–}1.82\text{ s}$ with zero residual acoustic events after $1.5\text{ s}$.

---

## 3. Brittle Fracture, Sub-Critical Crazing & 3D Voronoi Shards

### Griffith Energy Criterion, Progressive Crazing, and Flexural Snap

Annealed soda-lime glass undergoes negligible plastic yield prior to cleavage; failure is governed by Griffith's (1921) thermodynamic balance between elastic strain energy release rate $G$ and fracture surface energy $2\gamma_s$:

$$
G = \frac{\pi \sigma^2 a}{E'} \ge 2\gamma_s \implies \sigma_f = \frac{K_{Ic}}{\sqrt{\pi a}}
$$

where $K_{Ic} \approx 0.75\text{ MPa}\sqrt{\text{m}}$ is the Mode-I fracture toughness and $a$ is the dominant surface or internal flaw half-length. In `MujocoPhysicsEngine.ts`, sub-critical collisions ($v_{\text{impact}} \in [v_{\text{craze}}, v_{\text{shatter}})$) extend internal flaw radius, accumulated as normalized scalar damage $d \in [0, 0.92]$ anchored at the local impact point $\mathbf{p}_{\text{crack}} \in \mathbb{R}^3$.

Rather than treating `damage` as cosmetic, subsequent impacts evaluate degraded toughness $K_{Ic}(d) = K_{Ic,0}(1 - 0.36\,d)$:

$$
v_{\text{shatter}}(d) = v_{\text{shatter},0}\,(1 - 0.36\,d)
$$

Because free-fall impact velocity scales as $v = \sqrt{2 g H}$, the drop height onto the oak tabletop required to shatter a prism decreases quadratically with $(1 - 0.36\,d)^2$: empirically from $13.5\text{ cm}$ for pristine glass ($d = 0$) to $8.5\text{ cm}$ at $d = 0.55$ ($-37.0\%$) and $6.5\text{ cm}$ at $d = 0.85$ ($-51.9\%$), while gentle $1.0\text{ cm}$ placement drops remain below the crazing threshold ($0$ false damage).

Simultaneously, blocks bridged across outer supports under a central upper load undergo three-point Euler-Bernoulli bending over span $L_{\text{span}} = 60\text{ mm}$:

$$
M_{\text{bend}} = \frac{F_{\text{bend}} L_{\text{span}}}{4}, \qquad \sigma_b = \frac{M_{\text{bend}} (h/2)}{I_{zz}} = \frac{6 M_{\text{bend}}}{b h^2}
$$

When dynamic flexural stress exceeds the crazing-degraded threshold $\sigma_{\text{crit}}(d) = \sigma_{\text{crit},0}(1 - 0.48\,d)$ (gated safely above the $12.5\text{ N}$ static weight of an 18-course tower), flexural amplification triggers mid-span tensile snap (`|crackOrigin.x| < 0.18 dm`).

### Bounded Anisotropic 3D Voronoi Cleavage

When a block shatters at local point $\mathbf{p}_0$ under impact normal $\hat{\mathbf{n}}$, `src/physics/voronoiFracture.ts` partitions the exact 24-vertex beveled prism volume into $N_s \in [10, 24]$ convex polyhedral shards with $100\%$ volume conservation and zero inter-shard overlap. Seeds $\{\mathbf{s}_i\}_{i=1}^{N_s}$ are distributed with higher radial density near $\mathbf{p}_0$. To generate elongated radial splinters rather than isotropic foam cells while preserving planar faces (convexity), distance is measured under a symmetric positive-definite (SPD) metric tensor aligned with the impact basis $\{\hat{\mathbf{u}}_1, \hat{\mathbf{u}}_2, \hat{\mathbf{u}}_3\}$:

$$
\mathbf{A} = \sum_{k=1}^{3} \lambda_k (\hat{\mathbf{u}}_k \otimes \hat{\mathbf{u}}_k), \qquad (\lambda_1, \lambda_2, \lambda_3) = (0.38, 1.35, 2.25)
$$

Equating anisotropic squared distances $(\mathbf{x} - \mathbf{s}_i)^\top \mathbf{A}(\mathbf{x} - \mathbf{s}_i) = (\mathbf{x} - \mathbf{s}_j)^\top \mathbf{A}(\mathbf{x} - \mathbf{s}_j)$ cancels the quadratic $\mathbf{x}^\top \mathbf{A} \mathbf{x}$ terms identically, yielding an exact linear separating half-space:

$$
\mathbf{x}^\top \mathbf{A}(\mathbf{s}_j - \mathbf{s}_i) \le \frac{1}{2}\left(\mathbf{s}_j^\top \mathbf{A}\mathbf{s}_j - \mathbf{s}_i^\top \mathbf{A}\mathbf{s}_i\right)
$$

Sequential Sutherland-Hodgman 3D polygon clipping intersects the 18-plane beveled parent hull against each neighbor half-space, tagging faces inherited from the parent boundary (`isOuter = 1`) separately from newly created interior conchoidal cleavage planes (`isOuter = 0`).

### Exact Linear and Angular Momentum Projection & Asymmetric Shard Restitution

Because fracture is triggered during a soft Kelvin-Voigt collision (`solref[0] = 4.2 ms`), the parent block's center-of-mass velocity partway through contact penetration has not yet completed elastic rebound. `shatterBlock` reconstructs target pre-bounce normal rebound $v_{n,\text{target}} = e_{\text{fracture}} v_{\text{close}}$, overlays a subtle radial cleavage release velocity, and then projects out any net momentum drift across the shard ensemble $\{m_i, \mathbf{r}_i, \mathbf{I}_i, \mathbf{v}_i, \boldsymbol{\omega}_i\}$:

1. **Linear momentum projection**:

$$
\Delta \mathbf{v} = \mathbf{v}_0 - \frac{1}{M}\sum_{i=1}^{N_s} m_i \mathbf{v}_i, \qquad \mathbf{v}_i \leftarrow \mathbf{v}_i + \Delta\mathbf{v}
$$

2. **Angular momentum projection** via the $3 \times 3$ system inertia tensor about the parent center of mass $\mathbf{r}_0$:

$$
\mathbf{I}_{\text{tot}} = \sum_{i=1}^{N_s}\left[\mathbf{I}_i + m_i\left(\|\Delta\mathbf{r}_i\|^2 \mathbf{I}_3 - \Delta\mathbf{r}_i \otimes \Delta\mathbf{r}_i\right)\right], \qquad \Delta\boldsymbol{\omega} = \mathbf{I}_{\text{tot}}^{-1}\!\left(\mathbf{L}_0 - \sum_{i=1}^{N_s}\left(\mathbf{I}_i\boldsymbol{\omega}_i + m_i \Delta\mathbf{r}_i \times \mathbf{v}_i\right)\right)
$$

updating $\mathbf{v}_i \leftarrow \mathbf{v}_i + \Delta\boldsymbol{\omega} \times \Delta\mathbf{r}_i$ and $\boldsymbol{\omega}_i \leftarrow \boldsymbol{\omega}_i + \Delta\boldsymbol{\omega}$. Across randomized anvil-crush and free-flight fractures (`T3`), both linear and angular momentum errors remain $< 0.01\%$ ($< 1\%$ invariant gate).

Shards bind into a pre-allocated $512$-slot `sleep="init"` MuJoCo body pool (`16` convex hull archetypes, synchronized via `model.dof_M0` in `mujocoWorldCompat.ts` with a $r_{\text{gyr}} \ge 5\text{ mm}$ floor and $|\boldsymbol{\omega}| \le 55\text{ rad/s}$ gyroscopic cap so sub-gram splinters experience exact $1.000g$ gravity without explicit Euler $\boldsymbol{\omega} \times (\mathbf{I}\boldsymbol{\omega})$ blowup). To allow shards to bounce realistically in 3D on the oak table without turning shard piles into an underdamped spring mattress beneath intact $71.1\text{ g}$ blocks, the tabletop exposes a co-located secondary collision geom `oak_table_shard` (`MJ_CONTYPE_SHARD_TABLE = 0x20`, `priority = 1`, `shardSolref = [0.0060, 0.48]`, $\zeta = 0.48 \implies e \approx 0.18$) while shard-block and block-block contacts remain at `priority = 0` (`contactSolref = [0.0042, 1.12]`). During multi-block collapse cascades, staggering $> 2$ simultaneous Voronoi clip fractures across consecutive $1/600\text{ s}$ substeps keeps 95th-percentile step time at $14.22\text{ ms}$ ($4.97\text{ ms}$ mean).

### Contact-Support Free-Fall Gating, Circular Off-Rim Bitmasks & Passivity Fuzzing

Three invariants prevent subtle rigid-body artifacts across multi-stage stateful play:

1. **Upward contact-force gating (`d.qfrc_constraint`)**: Above the static tabletop ($y \ge 2.6\text{ cm}$), micro-jitter velocity damping is gated strictly by verified upward constraint support $f_{c,y} = \texttt{qfrc\_constraint}[\text{dof} + 1] > 2.2\,m_i$ rather than coordinate elevation or spawn age. Shards crossing a bounce apex ($v_y \approx 0$) or losing a supporting block later in a match experience $0.0$ artificial damping and accelerate immediately at $-9.81\text{ m/s}^2$.
2. **One-directional table `conaffinity` bitmask**: Because MuJoCo evaluates contact eligibility via symmetric OR (`(contype1 & conaffinity2) || (contype2 & conaffinity1)`), setting `MJ_CONAFFINITY_TABLE = 0x00` on `oak_table` makes collisions one-directional from piece `conaffinity` bit `0x01`: clearing bit `0x01` once a block or shard's horizontal radius crosses the $R = 43\text{ cm}$ circular table bullnose drops it cleanly past the rim without catching an invisible bounding cylinder.
3. **Stateful mechanical-energy passivity fuzzing (`scripts/fuzz-stateful-trajectories.mjs`)**: Across $520$ deterministic seeds $\times$ $80$ adversarial multi-action sequences (extractions, aborted pulls, mid-air releases, Sandbox shatters, and shard flicks), the fuzzer enables `mjENBL_ENERGY` and verifies that total mechanical energy $E_{\text{mech}} = E_{\text{pos}} + E_{\text{vel}}$ (`mj_energyPos` + `mj_energyVel`) never spontaneously increases across unforced settle windows, shrinking any violating trajectory via Delta Debugging (`ddmin`) to a minimal reproduction sequence.

---

## 4. Physical Optics of Cathedral Glass, Oak & Polariscopy

`WebGPURenderer.ts` executes a single-command-encoder `rgba16float` linear HDR pipeline (`4x` MSAA, `2048 x 2048` sun depth, optical-depth, and caustic maps, completing in $2.75\text{ ms}$ mean GPU time for an intact 18-layer tower and $3.64\text{ ms}$ mean / $7.43\text{ ms}$ p95 during a 454-shard collapse at $1920 \times 1200$ Retina on Apple Silicon).

### Exact Dielectric Fresnel, Total Internal Reflection & Cauchy Dispersion

Common real-time shaders approximate dielectric reflectance with Schlick's polynomial $F_0 + (1 - F_0)(1 - \cos\theta_i)^5$ evaluated on the incident cosine. For a ray exiting glass ($n_1 = 1.522$) into air ($n_2 = 1.0$), critical angle occurs at $\theta_c = \arcsin(1 / 1.522) = 41.07^\circ$: at $\theta_i = 40^\circ$, incident-cosine Schlick predicts $F = 0.0437$, underestimating true reflectance ($0.334$) by $7.6\times$.

`commonGlass.wgsl.ts` (`dielectricFresnel`) evaluates the exact unpolarized Maxwell-Fresnel equations across every air-glass, glass-air, internal crack-gap, and hollow seed-bubble interface:

$$
\sin^2\theta_t = \left(\frac{n_1}{n_2}\right)^2 (1 - \cos^2\theta_i), \qquad F(\theta_i) = \begin{cases} 1, & \sin^2\theta_t \ge 1 \;\text{(exact TIR)} \\[6pt] \dfrac{1}{2}\!\left[\left(\dfrac{n_1\cos\theta_i - n_2\cos\theta_t}{n_1\cos\theta_i + n_2\cos\theta_t}\right)^2 + \left(\dfrac{n_2\cos\theta_i - n_1\cos\theta_t}{n_2\cos\theta_i + n_1\cos\theta_t}\right)^2\right], & \sin^2\theta_t < 1 \end{cases}
$$

where $\cos\theta_t = \sqrt{1 - \sin^2\theta_t}$. Chromatic dispersion follows Cauchy's relation for soda-lime crown/cathedral glass ($V_d = (n_d - 1)/(n_F - n_C) \approx 58$), using $(n_R, n_G, n_B) = (1.514, 1.522, 1.533)$ across primary refraction, multi-block interior ray tracing ($\le 2$ interior blocks in `glassTower.wgsl.ts`), and 12-plane convex shard hull traversal (`glassShard.wgsl.ts`). Trapped spherical air bubbles (`evaluateSeedBubbles`) evaluate inverse Fresnel ($1.522 \to 1.0$), producing bright silver Total Internal Reflection annuli over impact parameter $b/R \in [1/1.522, 1] = [0.657, 1.0]$.

### Rolled Waterglass Potentials, Cames, and Internal Griffith Cracks

Hand-rolled cathedral glass exhibits gentle surface undulations without periodic grid seams or non-conservative normal curls. `evaluateVitrailPattern` defines a scalar $C^\infty$ heightfield $H(\mathbf{p}) = \sum_{m=1}^3 a_m \sin(\mathbf{k}_m \cdot \mathbf{p} + \phi_m)$ over three non-axis-aligned wavevectors, computing exact analytical surface slope $\nabla H$ ($\nabla \times \nabla H \equiv \mathbf{0}$) for Snell refraction and analytical Laplacian $\nabla^2 H = -\sum_m a_m \|\mathbf{k}_m\|^2 \sin(\mathbf{k}_m \cdot \mathbf{p} + \phi_m)$ for caustic focusing. Tracery motifs (`motifId` $\in \{1, 2, 3\}$) inset 3D half-round 60/40 tin-lead and copper-foil conductor came profiles strictly interior to the 12 diamond-polished outer bevels.

When a block accumulates crazing damage $d > 0$, `evaluateInternalCrackField` constructs block-local radial cleavage planes and a Hertzian conical impact shell around `crackOrigin` with Irwin linear-elastic crack-opening displacement $\delta_{\text{gap}}(r) \propto d\sqrt{1 - r/a} \in [140, 680]\text{ nm}$. Rays intersecting a cleavage plane evaluate glass-to-air-film Fresnel reflectance and Airy thin-film phase $\frac{4\pi \delta_{\text{gap}}\cos\theta}{\lambda}$, flashing mirror-bright when block wobble tilts the local cleavage normal past $\theta_c = 41.07^\circ$.

### Beer-Lambert Spectral Attenuation, Jacobian Sun Caustics & Quarter-Sawn Oak

Interior ray chords of length $\ell$ attenuate incident spectral radiance according to the Beer-Lambert law (Beer, 1852):

$$
\mathbf{T}(\ell) = (1 - F_{\text{in}})(1 - F_{\text{out}})\exp\!\left[-(\boldsymbol{\sigma}_{\text{dye}} + \boldsymbol{\sigma}_{\text{Fe}})\,\ell\right]
$$

where $\boldsymbol{\sigma}_{\text{Fe}} = (0.15, 0.045, 0.095)\text{ dm}^{-1}$ models residual $\text{Fe}^{2+}/\text{Fe}^{3+}$ iron-oxide absorption in soda-lime glass (producing the subtle sea-green tint along a $7.5\text{ cm}$ lengthwise sightline).

In `causticPass.wgsl.ts`, oblique solar rays refract through both intact blocks and all active 3D Voronoi shards (`vs_shard_caustic` / `fs_shard_caustic`, tracing exact entry-to-exit chords across `sh.planes[0..planeCount]`) onto the $y = 0$ tabletop plane. By conservation of radiative flux $\Phi = E_{\text{sun}} \, \mathrm{d}A_{\text{orig}} = E_{\text{table}} \, \mathrm{d}A_{\text{defl}}$, screen-space derivatives in the sun-orthographic rasterizer evaluate the inverse Jacobian determinant of the tabletop mapping $\mathbf{x}_{\text{orig}} \mapsto \mathbf{x}_{\text{defl}}$:

$$
I_{\text{caustic}} = \mathbf{T}(\ell) \cdot \operatorname{clamp}\!\left(\frac{|\det(\partial \mathbf{x}_{\text{orig}} / \partial \mathbf{u})|}{|\det(\partial \mathbf{x}_{\text{defl}} / \partial \mathbf{u})| + \varepsilon}, \; I_{\min}, \; I_{\max}\right)
$$

In `environmentPass.wgsl.ts`, `sampleContactHardeningCaustics` combines a $2048 \times 2048$ contact-hardening blocker search ($0.1\text{ mm}$ shadow contact width for Layer-0 blocks, $24\text{–}44\text{ mm}$ penumbra for Layer 12) with an alpha-channel shard-caustic tag (`uCausticTex.a`) so refracted Voronoi shard caustics project cleanly outside their un-deflected shadow footprint onto the ring-porous quarter-sawn oak tabletop (earlywood open-pore roughness modulation, medullary-ray chatoyancy, and solar-gated wood subsurface scatter).

### Maxwell-Brewster Stress-Optic Polariscopy

Isotropic glass becomes optically birefringent under mechanical stress (Brewster, 1815; Maxwell, 1853). A polarized ray traversing thickness $\ell$ through principal stress difference $\Delta\sigma = \sigma_1 - \sigma_2$ accumulates relative optical phase retardation:

$$
\Delta_{\text{nm}} = C \, (\sigma_1 - \sigma_2) \, \ell
$$

where $C \approx 2.6\text{ Brewsters}$ ($2.6 \times 10^{-12}\text{ Pa}^{-1}$) for soda-lime glass. Viewed through crossed dark-field polarizer and analyzer plates, transmitted spectral intensity at wavelength $\lambda \in \{635, 546, 455\}\text{ nm}$ follows:

$$
I(\lambda) = I_0(\lambda) \, \sin^2(2\theta_{\text{principal}}) \, \sin^2\!\left(\frac{\pi \Delta_{\text{nm}}}{\lambda}\right)
$$

Before MuJoCo's sleep system zeroes `d.ncon` on settled islands, `MujocoPhysicsEngine.ts` caches a 4-component local stress wrench per block (`photoelastic` $= (\sigma_{-X}, \sigma_0, \sigma_{+X}, \sigma_{\text{bend}})$), separating left, center, and right column bearing loads from three-point flexural tension. `glassTower.wgsl.ts` integrates this wrench at 3 sample points along each refracted interior ray chord, illuminating active load-bearing columns with first- and second-order interference fringes while unloaded loose blocks remain quiet. To prevent saturated Beer-Lambert dye absorption combined with analyzer extinction from turning unstressed prisms and shards into $< 1\%$ opaque matte-black silhouettes over sunlit oak, both `glassTower.wgsl.ts` and `glassShard.wgsl.ts` evaluate luma-neutralized smoked-crystal background refraction (`mix(0.30, 0.11, stressGate)` — $\approx 30\%$ neutral-density transmission on unloaded glass, deepening to $11\%$ inside stressed columns for high-contrast spectral $\sin^2(\pi\Delta/\lambda)$ troughs) alongside $50\text{–}72\%$ $s$-polarized Brewster oblique bevel/cleavage speculars and localized cleavage-edge strain glints on shattered Voronoi shards.

---

## 5. Computational Acoustics from Geometry & Elasticity

Every collision, tap-test, slide, wobble, and shard cascade in `src/audio/physicalModalModels.ts` and `src/audio/GlassAudioEngine.ts` is synthesized procedurally in Web Audio from physical elasticity parameters—zero recorded `.wav` or `.mp3` samples are used.

### Timoshenko Thick-Bar Eigenmodes & Viscoelastic Came Damping

A regulation $75 \times 25 \times 15\text{ mm}$ block has flatwise slenderness $L/h = 5.0$ and edgewise slenderness $L/b = 3.0$. Classical Euler-Bernoulli beam theory neglects rotary inertia and transverse shear deformation, overpredicting upper flexural overtones ($f_2/f_1 = (7.8532 / 4.7300)^2 = 2.7565$). Applying Timoshenko's (1921) coupled beam differential equation with Cowper's (1966) rectangular shear coefficient:

$$
\kappa_s = \frac{10(1 + \nu)}{12 + 11\nu} = 0.8461 \quad (\nu = 0.22)
$$

yields the exact dimensionless rotary-shear frequency reduction factor $\xi_n = f_n / f_{n,\text{EB}}$ for free-free root $\lambda_n \in \{4.7300, 7.8532, 10.9956\}$ and radius of gyration $r_g = h / \sqrt{12}$:

$$
\eta_n = \left(\frac{\lambda_n r_g}{L}\right)^2, \qquad B_n = 1 + \eta_n\!\left(1 + \frac{E_{\text{eff}}}{\kappa_s G_{\text{eff}}}\right), \qquad \xi_n = \sqrt{\frac{2}{B_n + \sqrt{B_n^2 - 4\frac{E_{\text{eff}}}{\kappa_s G_{\text{eff}}}\eta_n^2}}}
$$

Incorporating Reuss series compliance across a $3.4\%$ copper-foil / 60-40 $\text{Sn-Pb}$ solder came seam fraction ($E_{\text{glass}} = 72\text{ GPa}$, $\rho_{\text{glass}} = 2500\text{ kg/m}^3$), Saint-Venant rectangular torsion, and Rayleigh lateral-inertia longitudinal correction produces six physical modes per block:

| Mode Family & Order | Uncorrected $f_{\text{EB}}$ ($\text{Hz}$) | Corrected $f_k$ ($\text{Hz}$) | Factor $\xi_k$ | Quality $Q_k$ | $1/e$ Decay $\tau_k$ ($\text{ms}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Flatwise flexural $y$ ($n = 1$) | $3245.7$ | $2872.0$ | $0.8849$ | $249.3$ | $27.63$ |
| Saint-Venant torsional ($n = 1$) | $4114.5$ | $4052.8$ | $0.9850$ | $220.2$ | $17.29$ |
| Edgewise flexural $z$ ($n = 1$) | $5409.4$ | $4108.9$ | $0.7596$ | $219.1$ | $16.97$ |
| Flatwise flexural $y$ ($n = 2$) | $8946.8$ | $6806.2$ | $0.7607$ | $152.0$ | $7.11$ |
| Axial longitudinal ($n = 1$) | $7941.2$ | $7881.9$ | $0.9925$ | $172.0$ | $6.95$ |
| Flatwise flexural $y$ ($n = 3$) | $17539.3$ | $11399.1$ | $0.6499$ | $106.8$ | $2.98$ |

Notice $f_{y,2} / f_{y,1} = 2.3698 < 2.7565$, matching thick prismatic bar dispersion. Modal loss $\eta_{\text{loss}}(f) = Q^{-1}$ sums bulk glass friction, Zener (1937) transverse thermoelastic relaxation ($\tau_{\text{th}} = h^2 / (\pi^2 D_{\text{th}})$), frequency-dependent viscoelastic solder-came damping ($\propto f^{0.42}$, which damps harsh $> 10\text{ kHz}$ wineglass ring while keeping the $2.87\text{ kHz}$ cathedral clink clear), and structural crazing degradation ($E(d)/E_0 = 1 - 0.060\,d - 0.018\,d^2$, shifting $f_1$ down by $-1.54\%$ to $-3.98\%$ and splitting degenerate modes into beating doublets). Under column load $F_{\text{clamp}}$, a Winkler elastic foundation shift raises $f_1$ during tap-testing so players can hear bearing load by ear.

### Hertzian Contact Duration Spectral Excitation

Why does a gentle $5\text{ cm/s}$ block settle sound warm and muted while a $1.5\text{ m/s}$ collision sounds bright and metallic? By Hertzian contact mechanics (Hertz, 1882; Hunter, 1957), normal elastic impact between effective mass $m_{\text{eff}}$ and local bevel radius $R_{\text{eff}}$ across combined contact modulus $E^* = \left(\frac{1 - \nu_1^2}{E_1} + \frac{1 - \nu_2^2}{E_2}\right)^{-1}$ lasts for half-sine duration:

$$
\tau_c = 2.8684 \, m_{\text{eff}}^{2/5} R_{\text{eff}}^{-1/5} v_n^{-1/5} (E^*)^{-2/5}, \qquad f_c = \frac{1}{2\tau_c}
$$

Each modal partial $f_k$ is weighted by the Hertzian low-pass envelope $G(f_k, f_c) = \left[1 + (f_k / (1.35\,f_c))^4\right]^{-1}$:

- **Glass-on-glass** ($E^* = 37.83\text{ GPa}$, $m_{\text{eff}} = m/2$): a $0.05\text{ m/s}$ settle gives $\tau_c = 0.275\text{ ms}$ ($f_c = 1821\text{ Hz}$, attenuating the $11.4\text{ kHz}$ mode by $-42\text{ dB}$), whereas a $1.50\text{ m/s}$ strike shortens contact to $\tau_c = 0.139\text{ ms}$ ($f_c = 3596\text{ Hz}$, $> 7\times$ higher upper-partial ratio).
- **Glass-on-white-oak** ($E_{\text{oak},R} = 1.6\text{ GPa} \implies E^* = 1.81\text{ GPa}$, $m_{\text{eff}} = m$): radial wood compliance lengthens contact to $\tau_c = 0.619\text{–}1.222\text{ ms}$ ($f_c = 409\text{–}808\text{ Hz}$, $4.45\times$ lower cutoff than glass-on-glass).

### Orthotropic Oak Kirchhoff Plate Modes, Shard Similarity & Stick-Slip Singing

Impacts against the circular quarter-sawn white-oak tabletop ($R = 0.43\text{ m}$, thickness $h_p = 0.032\text{ m}$, $\rho = 750\text{ kg/m}^3$, orthotropic moduli $E_L = 12.0\text{ GPa}$, $E_R = 1.6\text{ GPa}$, $G_{LR} = 1.1\text{ GPa}$) excite Kirchhoff circular plate Bessel modes (Kirchhoff, 1850; Lekhnitskii, 1968; Leissa, 1969) using homogenized flexural rigidity $D_{\text{eff}} = \frac{1}{8}(3 D_x + 3 D_y + 2 D_{xy}) = 16,094\text{ N}\cdot\text{m}$ and Chaigne-Doutaut (1997) frequency-dependent wood internal friction:

$$
f_{mn} = \frac{\lambda_{mn}^2}{2\pi R^2}\sqrt{\frac{D_{\text{eff}}}{\rho h_p}} \implies (f_{01}, f_{11}, f_{21}, f_{02}) = (110.8,\, 200.6,\, 327.3,\, 508.1)\text{ Hz}
$$

weighted spatially by normalized radial strike coordinate $r / R \in [0, 1]$.

Post-shatter secondary 3D shard bounces (`ShardImpactEvent`) synthesize crystalline chattering via 3D elastodynamic similarity: for geometrically similar polyhedra of mass $m_i = \rho \ell_i^3$, natural frequency scales inversely with characteristic chord $\ell_i \propto m_i^{1/3}$:

$$
f_0(m_i) = f_{\text{ref}}\!\left(\frac{m_{\text{ref}}}{m_i}\right)^{1/3}, \qquad f_{\text{ref}}(1.0\text{ g}) = 5320\text{ Hz}
$$

mapping $15\text{ g}$ chunks to $2157\text{ Hz}$ and $0.2\text{ g}$ splinters to $9097\text{ Hz}$ (polyphony-capped at $12$ concurrent grains with stereo panning $\operatorname{clamp}(x_{\text{dm}} / 4.5, -0.85, 0.85)$). Finally, continuous block extraction blends broad low-$Q$ ($Q = 4.8$) airy micro-rasp on unloaded blocks into narrowband high-$Q$ ($Q \approx 35.7$) Stribeck stick-slip singing pulsed at relaxation rate $f_{\text{ss}} = 14 + v / d_{\text{ss}}$ when pulling heavily clamped load-bearing courses.

---

## 6. Counterfactual 2-Ply CEM AI (`mj_copyState`) & Chaos-Tested Internet Multiplayer

### Zero-Mutation 2-Ply Cross-Entropy Counterfactual Oracle (`Glazier AI`)

Because structural stability in a leaning frictional stack depends on non-local force redistribution, geometric heuristics cannot reliably distinguish a safe edge block from a catastrophic column support. In `MujocoPhysicsEngine.evaluateCounterfactualBlockCriticality`, the `Glazier AI` opponent (`Pass & Play -> AI`) executes a **2-ply counterfactual rollout** inside the background Web Worker using MuJoCo's native state-cloning API (fenced by `modelEpoch` so rebuilds never read stale rollouts):

1. **Zero-mutation state cloning**: Snapshot the live simulation state `(qpos, qvel, act, warmstart)` into a pre-allocated scratch `MjData` buffer via `mj_copyState(model, liveData, scratchData, SIG_INTEGRATION)` (`0` mutation of live `MjData`, preserving full 26-iteration Newton solver fidelity).
2. **Ply 1 — Cross-Entropy Method (CEM) pull-azimuth optimization**: Rather than testing only axial extraction, CEM refines pull azimuth $\psi \in [-\pi, \pi]$ (`optimalPullAzimuthRad`, `optimalPullDirXZ`) under the soft `<connect>` tether to minimize induced crest lean, column slip, and Stribeck breakaway impulse on subtly canted courses.
3. **Ply 2 — Adversarial top-course placement**: For top extraction candidates, the oracle simulates placing the extracted prism across open top-course slots with candidate lateral and yaw offsets (`adversarialPlacement`), maximizing a composite `twoPlyMasterScore01` combining self-survival margin (`selfSurvival01`) with asymmetric center-of-mass eccentricity left for the human opponent (`opponentDifficulty01`).

### Dual-Transport Host-Authoritative Multiplayer, `512`-Shard Cursor & Chaos Verification

`src/multiplayer/` enables two browsers anywhere on the public internet to play synchronized matches from `vitrail.andrewvoirol.com` with resilient P2P-to-edge failover:

- **Dual-Transport Connectivity & `<= 4.0 s` Automatic Failover (`VitrailMultiplayerClient.ts`)**: Peers exchange 5-character room codes (`?room=K7M4P`) over PeerJS (`&version=1.5.4`) WebSocket signaling and public STUN (`stun.cloudflare.com:3478`, `stun.l.google.com:19302`) to open two WebRTC SCTP `RTCDataChannel`s—`"ctrl"` (`ordered: true` reliable JSON for handshake, `roomSeed`, turn transitions, `shard_spawn` convex plane hulls, and deduplicated `audio_batch` events) and `"state"` (`ordered: false, maxRetransmits: 0` binary `ArrayBuffer` snapshots at $30\text{ Hz}$). Whenever WebRTC ICE stalls beyond $2.8\text{ s}$ ($\le 4.0\text{ s}$ ceiling) or drops mid-drag under symmetric NATs, traffic fails over automatically to a hibernatable SQLite Cloudflare Durable Object WebSocket relay (`workers/vitrail-rooms/src/index.ts`, guarded by a non-inhibiting `Origin` allowlist, `16 KB` `0x564d` binary / `512 KB` JSON frame validation, and `1008` 3rd-peer occupancy lock).
- **Deterministic Seeding, `< 4 KB` Wire Frames & Round-Robin `512`-Shard Replication (`protocol.ts`)**: Worker Host, synchronous Host, and Guest (`MultiplayerGuestWorld.ts`) evaluate the identical sequential `buildDeterministicTowerMaterials` PRNG stream (`mulberry32((roomSeed ^ Math.imul(towerLayers, 0x9e3779b9)) >>> 0)`), guaranteeing `54 / 54` bit-identical stained-glass swatches, tracery motifs, and waterglass ripple phases. Wire frames (`MP_PROTOCOL_MAGIC = 0x564d`) pack a $140\text{ B}$ header, $32\text{ B/block}$ (`Float32x3` position, $w \ge 0$ hemisphere `Int16x4` quaternion, `Uint16x4` 4-component Brewster photoelastic stress wrench, damage, and flags), and $16\text{ B/shard}$ ($\le 120$ shards/packet: **$1,868\text{ B}$** intact, **$3,908\text{ B} < 4\text{ KB}$** peak). When a collapse spawns $> 120$ Voronoi shards (up to `512`), `SerializedShardMeta` seeds exact `initialPosition`/`initialRotation` on spawn while a round-robin `mpShardCursor` cycles $120$-shard slices across consecutive $30\text{ Hz}$ packets for $1:1$ full-pile convergence.
- **Monotonic Rematch/Audio Fencing & Antipodal-Safe Slerp**: Host state sequence numbers (`mpStateSeq`) increment monotonically across rematches and Guest rejects `frame.seq <= lastAcceptedSeq` *ahead* of `roomSeed` checks so delayed UDP packets cannot roll back a fresh rematch; Host audio taps use a top-level re-entrancy guard (`hostAudioTapDepth`) plus a `128`-batch `batchSeq` ring (`<= 25 Hz` collapse transient cap) for exact single-emission audio parity. On the Guest, $30\text{ Hz}$ snapshots render smoothly at $120\text{ Hz}$ via shortest-arc `quatSlerpHemisphere` ($\alpha = 1 - \exp(-\Delta t / \tau)$, $\tau = 36\text{ ms}$):

$$
\mathbf{q}_1^* = \begin{cases} -\mathbf{q}_1, & \mathbf{q}_0 \cdot \mathbf{q}_1 < 0 \\ +\mathbf{q}_1, & \mathbf{q}_0 \cdot \mathbf{q}_1 \ge 0 \end{cases}, \qquad \operatorname{slerp}(\mathbf{q}_0, \mathbf{q}_1^*, \alpha) = \frac{\sin((1-\alpha)\Omega)\,\mathbf{q}_0 + \sin(\alpha\Omega)\,\mathbf{q}_1^*}{\sin\Omega}
$$

where $\Omega = \arccos(\mathbf{q}_0 \cdot \mathbf{q}_1^*)$. Occluded Host tabs step via a $30\text{ Hz}$ background pump (`startHostBackgroundPump`), `sessionStorage` (`vitrail_mp_tab_session_v1`) reclaims Host or Guest roles across `F5` reloads, and `scripts/chaos-multiplayer-sim.mjs` verifies all invariants across $100$ seeded matches under simultaneous $8\%$ baseline packet drop, $65\%$ burst loss, $120 \pm 45\text{ ms}$ jitter/reordering, and mid-drag WebRTC-to-WS failover ($< 2.0\text{ mm}$ Guest convergence within $250\text{ ms}$).

---

## 7. Primary References

1. **Beer, A.** (1852). "Bestimmung der Absorption des rothen Lichts in farbigen Flüssigkeiten." *Annalen der Physik*, 162(5), 78–88. DOI: `10.1002/andp.18521620505`
2. **Brewster, D.** (1815). "On the effects of simple pressure in producing that species of crystallization which forms two oppositely polarised images, and exhibits the complementary colours by polarised light." *Philosophical Transactions of the Royal Society of London*, 105, 60–64. DOI: `10.1098/rstl.1815.0006`
3. **Chaigne, A. J., & Doutaut, V.** (1997). "Numerical simulations of xylophones. I. Time-domain modeling of the vibrating bars." *The Journal of the Acoustical Society of America*, 101(1), 539–557. DOI: `10.1121/1.418117`
4. **Cowper, G. R.** (1966). "The shear coefficient in Timoshenko's beam theory." *Journal of Applied Mechanics*, 33(2), 335–340. DOI: `10.1115/1.3625046`
5. **Gauss, C. F.** (1829). "Über ein neues allgemeines Grundgesetz der Mechanik." *Journal für die reine und angewandte Mathematik*, 4, 232–235. DOI: `10.1515/crll.1829.4.232`
6. **Griffith, A. A.** (1921). "The phenomena of rupture and flow in solids." *Philosophical Transactions of the Royal Society of London. Series A*, 221, 163–198. DOI: `10.1098/rsta.1921.0006`
7. **Hertz, H.** (1882). "Ueber die Berührung fester elastischer Körper." *Journal für die reine und angewandte Mathematik*, 92, 156–171. DOI: `10.1515/crll.1882.92.156`
8. **Housner, G. W.** (1963). "The behavior of inverted pendulum structures during earthquakes." *Bulletin of the Seismological Society of America*, 53(2), 403–417.
9. **Hunter, S. C.** (1957). "Energy absorbed by elastic waves during impact." *Journal of the Mechanics and Physics of Solids*, 5(3), 162–171. DOI: `10.1016/0022-5096(57)90002-9`
10. **Kirchhoff, G.** (1850). "Über das Gleichgewicht und die Bewegung einer elastischen Scheibe." *Journal für die reine und angewandte Mathematik*, 40, 51–88. DOI: `10.1515/crll.1850.40.51`
11. **Leissa, A. W.** (1969). *Vibration of Plates*. NASA SP-160, National Aeronautics and Space Administration, Washington, D.C., 353 pp.
12. **Lekhnitskii, S. G.** (1968). *Anisotropic Plates* (trans. S. W. Tsai & T. Cheron). Gordon and Breach Science Publishers, New York, 534 pp.
13. **Maxwell, J. C.** (1853). "On the equilibrium of elastic solids." *Transactions of the Royal Society of Edinburgh*, 20(1), 87–120. DOI: `10.1017/S0080456800033044`
14. **Stribeck, R.** (1902). "Die wesentlichen Eigenschaften der Gleit- und Rollenlager." *Zeitschrift des Vereines Deutscher Ingenieure*, 46(37–39), 1341–1348, 1432–1438, 1463–1470.
15. **Timoshenko, S. P.** (1921). "On the correction for shear of the differential equation for transverse vibrations of prismatic bars." *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, Series 6, 41(245), 744–746. DOI: `10.1080/14786442108636264`
16. **Todorov, E., Erez, T., & Tassa, Y.** (2012). "MuJoCo: A physics engine for model-based control." *2012 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 5026–5033. DOI: `10.1109/IROS.2012.6386109`
17. **Zener, C.** (1937). "Internal friction in solids. I. Theory of internal friction in reeds." *Physical Review*, 52(3), 230–235. DOI: `10.1103/PhysRev.52.230`
