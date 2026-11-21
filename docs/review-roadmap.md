# Review roadmap (not done yet)

Findings from a code review that were **not** changed, because they alter visible output and need
real photographs, SMPL-X weights or a GPU to validate. Roughly ordered by expected impact.

1. **Input quality.** The four views are generated panels about 534 px tall (subject ~150 px wide)
   stretched to a 4K atlas. Use higher-resolution, mutually consistent views or super-resolve
   before projection. Use the original photo as a held-out check for the body fit (fit pose only,
   fixed shape, compare silhouette IoU); it is not used by the geometry fit.
2. **Segmentation.** `prepare.py` assigns labels from absolute fractions of the image height and RGB
   thresholds; "skin" means "foreground not claimed by another class". Replace with a human-parsing
   model and a matting model.
3. **Texture.** Skin, hair and lace are procedural. Faces choose one source view (arg-max) with no
   blending, so seams are expected. Bake per-texel in UV space with angle/edge weights and
   multi-band blending; pack with xatlas instead of one near-full-image tile per source.
4. **Projection pose.** `geometry.py` poses garment vertices with the nearest body vertex's
   transform while export uses the blended skin weights `W`; they disagree near the crotch.
5. **Face.** 51 landmarks with piecewise-affine warping; use dense landmarks or a FLAME fit and a
   single best source view.
6. **Fit.** Replace the vertex-sampled silhouette terms with a differentiable soft silhouette;
   add a normal term; perspective camera (pitch/focal); compose pose offsets as rotations rather
   than adding axis-angle vectors; add hands/feet keypoints; camera init uses 1.77 m while the
   prior is 1.695 m.
7. **Print.** The voxel field is ~4 mm at body scale (~0.5 mm at 200 mm print height); the depth
   envelope fills front/back cavities; no wall-thickness check or hollowing.
8. **Engineering.** Stages are import-time scripts; split into functions. Loosen exact dependency
   pins; add a lock file and a licence.
