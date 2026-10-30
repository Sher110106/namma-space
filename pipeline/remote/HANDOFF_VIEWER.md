# Handoff to the viewer builder (from the integrator, 30 Oct 11:20 IST)

1. **R9 and R10 fail on every real run, so `ns run` ends with 30.** On `runs/rh-honka` (a scene with no `places`):
   - R9: "start card mentions search, Round 2 preview and measure; help lists / J G Space K P; ?place=door&route=1 ->
     route drawn (null); 0 console errors" (pass false: the route is null because the scene has no place `door`).
   - R10: "search button not visible on the phone".
   R1-R5 and R8 already report "not run: scene has no places". Suggested: R9's route part and R10 skip (pass null)
   when the scene has no places, like R1-R5. `ns run` treats any failed viewer check as exit 30 ("bundle exists,
   needs review"), so today every run, including the user's room, will end 30 for this reason alone.
   Evidence: `runs/rh-honka/90_verify/verify.json` (25 passed, 2 failed, 6 not run).
2. Nothing else blocks: `bundle.py` passed 78/78 checks on rh-honka and rh-vrroom; H1-H9 passed on rh-honka.
