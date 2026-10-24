"""go2_teacher — privileged (scandot) wall-plant teacher config.

The single configuration used by the release. It is the flattened form of what
used to be a two-level chain: the former ``Go2TeacherCfg`` body with the
``Go2TeacherCfg`` overrides folded in, now inheriting ``Go2RoughCfg``
directly. Sites where the swimmer layer overrode a teacher value are marked
``[swimmer]``.

The three families the swimmer layer changed, for the record:

  1. rewards.wococo_wall_press_force_threshold_by_level = [10, 10, 35, 45, 50]
     — a per-level ramp replacing the flat 10 N wall-press gate (harder to
     register a wall plant on the upper curriculum rows).

  2. terrain foothold-GOAL LOCATIONS at the e68094d ``go2_cliffside`` values:
     the hand-authored 30-80 deg foothold table, the 5 cm on-face z-lift (vs
     0.15 for the default foothold goals), and the cliffside landing-foot / rear-foothold
     geometry. The wall-angle range, plate, run-up, and pit offset already
     matched cliffside, so they were never overridden.

  3. a compact 4-waypoint goal circuit (``flyball_swimmer_goal_layout``):
     g0 between-hurdles midpoint -> g1 wall -> g2 between-hurdles midpoint ->
     g3 final return goal (g4..g7 padded). The takeoff and landing waypoints
     are dropped, so the wall goal moves from index 2 to 1 and the return phase
     from 4 to 2; both are declared via ``flyball_wall_goal_idx`` /
     ``flyball_return_goal_idx``, which every env19 goal gate in
     ``legged_robot.py`` reads. Goal promote/demote fractions are re-anchored to
     the shorter circuit.

The per-angle acceptance radii / kappas (``wococo_per_angle_radii_low``,
``wococo_foot_region_radius``, ``wococo_landing_foot_region_radius``, ...) were
INTENTIONALLY left at the teacher values by the swimmer layer — only the goal
*positions* moved. They are unchanged here.

    python train.py --exptid <id> --task go2_teacher --device cuda:0
"""

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.go2.go2_config import Go2RoughCfg, Go2RoughCfgPPO


# ACTIVE [swimmer]. Hand-tuned, row every 5 deg 30-80. final_on_face=True, so
# the z-offset lift and rear +y shift are skipped for it.
_FOOTHOLD_TABLE_MANUAL_30_80 = (
    LEGGED_GYM_ROOT_DIR + "/resources/footholds/foothold_table_manual_30_80.npz"
)

# Opt-in alternative (the pre-swimmer default): rows at 20/30/40/50 deg, pre-lift
# (z-offset and rear +y shift apply on top), end-clamped above 50 deg. Selecting it
# means reverting flyball_foothold_z_offset_m to 0.15 and the rear +y shift to 0.00.
_FOOTHOLD_GOALS_DEFAULT = (
    LEGGED_GYM_ROOT_DIR + "/resources/footholds/foothold_table.npz"
)


class Go2TeacherCfg( Go2RoughCfg ):

    class sim( Go2RoughCfg.sim ):
        class physx( Go2RoughCfg.sim.physx ):
            default_buffer_size_multiplier = 10   # wall base spikes collision pairs

    class env( Go2RoughCfg.env ):
        episode_length_s = 20
        roll_term_deg = 95.0    # banked wall plant needs large roll tolerance
        next_goal_threshold       = 0.2
        terminate_on_lateral_deviation = True
        lateral_deviation_term_m       = 1.5

    class commands( Go2RoughCfg.commands ):
        # Per episode: STANDSTILL (random 0-4 s) -> RUN -> HARD STOP.
        wococo_bimodal_cmd   = True
        # Force the bimodal wall-command sampler ON regardless of the wococo_stage
        # reward scale (otherwise the nostage ablation falls through to the generic
        # continuous U[lin_vel_x] draw, e.g. 0.38 m/s). No-op for non-wall classes.
        wococo_bimodal_cmd_force = True
        wococo_bimodal_env_classes = [19, 15]
        initial_standstill_max_s = 4.0
        wococo_zero_cmd_prob     = 0.0
        resample_only_on_reset = True
        class ranges( Go2RoughCfg.commands.ranges ):
            lin_vel_x = [0.0, 1.0]
        class max_ranges( Go2RoughCfg.commands.max_ranges ):
            lin_vel_x = [1.0, 1.0]

    class control( Go2RoughCfg.control ):
        stiffness = {'joint': 35.}   # sim2real-validated
        damping   = {'joint': 0.6}

    class domain_rand( Go2RoughCfg.domain_rand ):
        # PD-gain mismatch randomized alongside the observed motor-strength mismatch.
        randomize_pd_gain          = True
        randomize_motor_offset     = True
        randomize_restitution      = True
        restitution_range          = [0.0, 0.5]
        randomize_friction         = True
        friction_range             = [0.6, 2.0]
        # Per-env, per-joint PD gain multipliers.
        kp_range                   = [0.95, 1.05]
        kd_range                   = [0.95, 1.05]
        randomize_base_mass        = True
        base_mass_range            = None
        added_mass_range           = [0.0, 3.0]
        randomize_base_com         = True
        added_com_range            = [-0.2, 0.2]

        # Disturbance settings (pushes remain disabled).
        push_robots                = False
        push_interval_s            = 4
        max_push_vel_xy            = 0.25
        max_push_vel_z             = 0.0
        max_push_ang_vel           = 0.25

        randomize_motor            = True
        motor_strength_range       = [0.8, 1.2]

        # Fixed-delay model: no per-env action/proprio latency DR.
        randomize_action_latency   = False
        randomize_proprio_latency  = False
        action_delay               = True
        action_delay_latch_mean_level = 6.0
        action_delay_latch_step       = None
        action_delay_ramp_steps       = 0
        randomize_contact_threshold = False

        # Fleet-mean DR ramp: the ceiling reaches full DR at mean level 6, and each
        # reset samples a per-env scale below it. Resumes are pre-armed by helpers.py.
        dr_scale_range             = [0.0, 1.0]
        dr_lock_to_terrain_level   = True
        dr_ramp_use_mean_level     = True
        dr_scale_sample_per_env    = True
        dr_ramp_full_at_mean_level = 6.0
        dr_ramp_arm_below_mean_level = 0.5
        dr_ramp_start_armed        = False

    class noise(Go2RoughCfg.noise):
        # The DR ramp above scales these proprio/height noise samples by dr_scale;
        # privileged state stays clean. Depth DR is deliberately not overridden here.
        add_noise = True
        noise_level = 1.0

        class noise_scales(Go2RoughCfg.noise.noise_scales):
            rotation = 0.05

    class asset( Go2RoughCfg.asset ):
        penalize_contacts_on        = ["thigh", "calf", "Head"]
        terminate_after_contacts_on = ["base", "Head"]

    class depth( Go2RoughCfg.depth ):
        # helpers.py reads the camera terrain rewrites off cfg.depth, NOT
        # cfg.terrain, so these have to live here to take effect. The teacher
        # owns its 8 m tiles at 5 cm and the IsaacLab rough mix; the stock
        # camera rewrite would coarsen cells to horizontal_scale_camera (0.1)
        # and inject parkour_* slots, which overruns demo_terrain's 8 m layout.
        override_terrain_mix_for_camera  = False
        override_terrain_grid_for_camera = False

    class terrain( Go2RoughCfg.terrain ):
        # measured_points_* intentionally inherited: the base grid is now 17 x 17
        # over +-2 m, aligned with the camera-reconstructed scan grid.
        # The height obs no longer comes from the privileged query, so the
        # scandot post-processing that only shaped that query is switched off.
        scandot_maxpool = False
        scandot_occlusion_mask = False

        curriculum  = True
        no_demotion = False

        # IsaacLab rough.py layout: 8 x 8 m tiles, 10 curriculum rows, 20 type
        # columns. 20 columns is what makes the 6-way split land on even
        # boundaries: curiculum() sets choice = j / num_cols, so each 0.2-wide
        # slot below owns exactly 4 columns.
        num_rows       = 10
        num_cols       = 20
        terrain_length = 8.0
        terrain_width  = 8.0

        # 5 cm cells. The old 1 cm existed only to resolve the flyball 1 cm bars;
        # on 8 m tiles it would need a 5x taller/wider height field for nothing.
        # The map is then (10*8 + 2*5) x (20*8 + 2*5) m = 1800 x 3400 px.
        horizontal_scale = 0.05
        border_size      = 5
        hf2mesh_method   = "fast"
        max_error        = 1.0

        # ---- IsaacLab rough.py mix (ROUGH_TERRAINS_CFG.sub_terrains) --------
        # Slot i is the branch for proportions[i]. A slope/stair inverted pair
        # shares one slot: make_terrain splits it in half and flips the sign.
        #   slot 0 -> idx 0  HfPyramidSlopedTerrainCfg            0.10
        #          -> idx 1  HfInvertedPyramidSlopedTerrainCfg    0.10
        #   slot 3 -> idx 5  MeshInvertedPyramidStairsTerrainCfg  0.20
        #   slot 4 -> idx 4  MeshPyramidStairsTerrainCfg          0.20
        #   slot 5 -> idx 6  MeshRandomGridTerrainCfg             0.20
        #   slot 8 -> idx 9  HfRandomUniformTerrainCfg            0.20
        terrain_dict = {"smooth slope": 0.2,
                        "rough slope up": 0.0,
                        "rough slope down": 0.0,
                        "rough stairs up": 0.2,
                        "rough stairs down": 0.2,
                        "discrete": 0.2,
                        "stepping stones": 0.0,
                        "gaps": 0.0,
                        "smooth flat": 0.2,
                        "parkour_stones": 0.0,
                        "wall": 0.0,
                        "platform": 0.0,
                        "large stairs up": 0.0,
                        "large stairs down": 0.0,
                        "parkour": 0.0,
                        "two_hurdles_wall_turn": 0.0,
                        "parkour_flat": 0.0,
                        "parkour_step": 0.0,
                        "flyball_box": 0.0,
                        "demo": 0.0,
                        }
        terrain_proportions = list(terrain_dict.values())

        # Roughness amplitude added ON TOP of stairs/slopes by add_roughness.
        # Kept small (2-6 cm): it is a surface break-up for sim2real, not the
        # terrain itself. The pure-noise tile gets its own 2-10 cm range from
        # HfRandomUniformTerrainCfg's noise_range inside make_terrain.
        height = [0.02, 0.06]
        flyball_roughness_zero_prob = 0.5

        # ---- below is flyball-only and inert with the mix above ------------

        flyball_add_hurdle                  = True
        flyball_add_second_hurdle           = True
        # Rows 0-3 are hurdle-free. Rows 4-9 interpolate 7 cm -> 12 cm.
        flyball_single_hurdle_height_range  = (0.07, 0.12)
        flyball_single_hurdle_zero_below_level = 4
        flyball_single_hurdle_height_max      = 0.12
        # Bar presence and height DECORRELATED from row difficulty: lerping bar height
        # on the difficulty variable taught the arms to read bar height as a difficulty
        # cue. On rows >= 4 each slot is independently EMPTY with probability 0.25 or a bar with height uniform over the
        # (0.07, 0.12) range — never below the 7 cm deployment minimum, and
        # never a function of the row. The jitter is zeroed here so a revert to the
        # lerp cannot reintroduce sub-minimum bars.
        flyball_hurdle_height_decorrelate   = True
        flyball_hurdle_zero_prob            = 0.25
        flyball_hurdle_height_jitter_m      = 0.0
        flyball_single_hurdle_thickness      = 0.01
        flyball_single_hurdle_thickness_range = None
        flyball_single_hurdle_thickness_range_uniform = False
        flyball_hurdle_thickness_jitter_m     = 0.0
        flyball_single_hurdle_thickness_max   = 0.01
        flyball_single_hurdle_width           = 1.20
        flyball_single_hurdle_width_range     = (1.00, 1.30)
        # Clearance parabola: 20 cm apex at hurdle centre, roots 10 cm each side.
        flyball_hurdle_clear_half_span         = 0.10
        flyball_hurdle_clear_box_height        = 0.17
        flyball_hurdle_roughness_height       = (0.0, 0.01)
        flyball_hurdle_jitter_scale_with_difficulty = True

        flyball_spawn_x                       = 1.0
        flyball_hurdle_front_from_spawn       = 1.60
        flyball_hurdle_front_from_spawn_jitter = 0.30
        flyball_second_hurdle_offset          = 2.00
        flyball_second_hurdle_offset_jitter   = 0.30
        flyball_wall_to_hurdle_rear           = 2.00
        flyball_wall_to_hurdle_rear_jitter    = 0.30
        # Hard floor on the JITTERED hurdle-2-rear -> wall-toe gap: the -0.30 tail put
        # the last hurdle 1.70 m from the wall, too little run-up for plant speed.
        # Floors rather than shifts: ~half the tiles sit at 2.00 m, the rest 2.00-2.30 m.
        flyball_wall_to_hurdle_rear_min       = 2.00
        flyball_post_hurdle_goal_distance     = 0.0
        flyball_randomize_return_goal_x        = True
        flyball_plate_width                 = 1.3
        flyball_plate_length                = 1.1   # along-slope (hypotenuse)
        flyball_runup_dist_range            = (2.25, 2.25)  # hurdle-free fallback only
        flyball_runup_jitter                = 0.0
        flyball_pit_height_offset_range     =  (0,0)
        flyball_wall_angle_deg_range        = (30.0, 75.0)  # +-5 jitter -> true max 80
        flyball_wall_angle_jitter_deg       = 5.0

        # foothold GOAL LOCATIONS [swimmer]: hand-authored per-5deg table (30-80 deg);
        # flyball_wall.py skips the z-offset clamp / >50deg handling for it.
        flyball_foothold_table_path         = _FOOTHOLD_TABLE_MANUAL_30_80
        # [swimmer] 0.05 (was 0.15 for the default foothold goals). Raises every WALL
        # foot target by this much in z (up-slope by dz/sin(theta), so the target stays
        # on the face and x moves toward the wall by dz/tan(theta)). Inert for a
        # final_on_face table such as the one above.
        flyball_foothold_z_offset_m         = 0.05

        # Front-foot floor (landing) targets used by the post-turn landing stage.
        flyball_landing_foot_standoff       = 0.80   # [swimmer] goals 5 cm farther from the wall
        flyball_landing_foot_y_mid          = 0.00   # [swimmer] center FL/FR goals on the corridor y
        flyball_landing_foot_stance_half    = 0.10
        flyball_landing_foot_x_stagger      = 0.10   # [swimmer] was 0.00
        # Rear wall footholds: shift +5 cm in +y on walls steeper than 50 deg.
        flyball_rear_foothold_positive_y_shift_m  = 0.05   # [swimmer] was 0.00
        flyball_rear_foothold_shift_min_angle_deg = 50.0   # [swimmer]

        # Compact 4-waypoint goal circuit [swimmer]: g0 between-hurdles midpoint,
        # g1 wall contact, g2 that midpoint again on the return leg, g3 final return
        # goal (g4..g7 padded). Takeoff and landing waypoints are dropped.
        flyball_swimmer_goal_layout = True
        # Every env19 goal gate in legged_robot.py reads these two indices, so
        # they MUST track the layout above (wall 2 -> 1, return phase 4 -> 2).
        flyball_wall_goal_idx   = 1
        flyball_return_goal_idx = 2

        max_init_terrain_level = 5
        resample_at_max_level  = True
        resample_min_level     = 4
        # [swimmer] Goal curriculum re-anchored to the 4-waypoint circuit. max_goal_idx
        # means: 1 = passed g0, 2 = plant complete, 3 = back between the hurdles,
        # >=4 = finished. Promote at "plant done + one waypoint", demote at "no plant".
        promote_goal_frac      = 0.375   # int(0.375*8) = 3
        demote_goal_frac       = 0.25    # int(0.25*8)  = 2

    class rewards( Go2RoughCfg.rewards ):
        base_height_target = 0.25
        max_contact_force = 110.0   # [N] per-foot resultant-force threshold
        exempt_hurdle_contact_from_clip   = True
        exempt_hurdle_clearance_from_clip = False
        exempt_wococo_contact_pts_from_clip = True   # n_wrong must reach negative
        # Clip only below level 2, so regularization bites once past cold start.
        only_positive_rewards_exempt_high_levels = True
        only_positive_rewards_clip_below_level    = 2

        # Arms at a fixed iteration, not a level: a level trigger feeds back (the
        # penalty demotes the fleet, and demotion then disarms it).
        hurdle_clearance_arm_step             = 2000 * 24
        hurdle_clearance_apex_full_mean_level = None
        hurdle_clearance_arm_mean_level       = 0.0
        hurdle_clearance_ramp_delay_steps     = 0
        hurdle_clearance_ramp_steps           = 4000 * 24
        hurdle_clearance_per_env_draw        = True
        hurdle_pair_sym_sigma_rad             = 0.40   # summed mirror error per pair

        exempt_penalties_at_wall_plant = False
        wococo_roll_exempt_deg              = 100.0
        wococo_roll_exempt_post_sentinel_s  = 2.0

        progressive_tracking_gating       = False
        gate_tracking_during_wococo_plant = True
        tracking_goal_vel_unmask_stage    = 2   # vel re-enables at the LANDING
        tracking_yaw_unmask_stage         = 1   # yaw re-enables at the REAR_PLANT
        tracking_fadeout_episodes         = 0.0
        gate_tracking_yaw_at_standstill   = True
        standstill_dof_error_scale        = 20.0
        standstill_dof_error_ramp_s       = 1.0
        standstill_orientation_sigma      = 0.05

        # Reaching the final goal zeroes the command and holds the episode open instead
        # of resetting, so braking to a standstill is trained.
        enable_final_goal_hard_stop = True
        final_goal_stop_s           = 5.0

        # P4L stages: 0 FRONT_PLANT -> 1 REAR_PLANT -> 2 LANDING -> 3 sentinel.
        wococo_design               = 'P4L'
        wococo_plant_complete_stage = 2   # wall goal advances entering the landing
        wococo_landing_foot_region_radius = 0.60
        wococo_landing_contact_force_threshold = 5.0
        wococo_landing_roll_tol_deg = 25.0
        # Under wococo_landing_posture_only this is the tolerance from the RETURN
        # BEARING (of flyball_return_goal_idx), not from a hard-coded 180 deg as in the
        # foothold landing, which reads the same knob as "within X deg of 180" — so
        # reverting posture_only tightens that gate from the historic 45/70 to 40.
        wococo_landing_yaw_tol_deg  = 40.0
        wococo_landing_F_task_sigma = 1.5
        wococo_landing_stagger_window_steps = 30   # lets the pair land sequentially
        wococo_goal_advance_mode    = 'plant_stage'
        wococo_stage_reward_mode    = 'oneshot_linear'
        wococo_n_wrong_weight       = 1.0

        # Anneal the one-shot stage bonus to zero on FLEET-MEAN terrain level: it buys
        # its faster curriculum onset below level 5, and kept live afterwards it keeps
        # compressing the plant trigger against distribution-specific cues. Ramps
        # 25 -> 0 over fleet-mean level 5 -> 7 and LATCHES, so demotions and resumes
        # cannot resurrect it. The FSM is unaffected; start = None disables.
        wococo_stage_anneal_start_mean_level = 5.0
        wococo_stage_anneal_end_mean_level   = 7.0
        wococo_stage_anneal_final_frac       = 0.0

        # Posture-only LANDING (stage 2), ON: stage 2 stops regulating foot contacts
        # entirely (no footholds, no n_wrong, n_con = 0) and becomes a pure BASE-STATE
        # gate on the return posture: |roll| < roll_tol, |pitch| < pitch_tol,
        # |yaw - bearing(return goal)| < yaw_tol, height in [h_lo, h_hi], and
        # return-projected speed >= frac * cmd, all held for wococo_landing_n_persist
        # steps (that hold is what makes the gate unreachable in mid-air).
        # Turning this OFF restores the foothold landing but
        # ALSO requires reverting wococo_landing_n_persist (-> 1) and the
        # wococo_landing_foot_target scale (-> 25.0) below.
        wococo_landing_posture_only  = True
        wococo_landing_pitch_tol_deg = 35.0   # a 1 m/s trot pitches +/-5-10 deg
        wococo_landing_pitch_weight  = 1.0    # pitch weight in the dense err
        wococo_landing_height_weight = 10.0   # metres of height shortfall -> rad
        wococo_landing_h_lo          = 0.15   # below = trunk down (belly-slide)
        # h_hi bounds ballistic dwell: a body entering the band from above holds
        # it for at most sqrt(2*(h_hi-h_lo)/g) = 10.1 steps at [0.15, 0.35].
        # RAISING h_hi WITHOUT RAISING n_persist reopens mid-air completion.
        wococo_landing_h_hi          = 0.35
        wococo_landing_speed_frac    = 0.8    # fraction of commands[:, 0]
        # Stage-2-only multiplier on wococo_base_orient (stages 0/1 unaffected).
        # 1.5 = +50%: the ONLY dense signal for roll/pitch/base-height at the landing,
        # carrying a 15-step pause-on-error hold. Raising it also raises the landing's
        # draw on wococo_income_episode_cap -- the two must move together.
        wococo_landing_base_orient_gain = 1.5

        # Per-wall-angle radii OFF: every consumer falls back to its flat scalar below.
        # A dict re-enables it (interpolated per env angle, clamped outside 30-80 deg).
        wococo_per_angle_radii = None
        # Cold-start gates, restricted to bootstrap levels 0-3; above level 3 the scalar mature-row tolerances hold.
        wococo_per_angle_radii_low_below_level = 4
        wococo_per_angle_radii_low = {
            30.0: (0.500, 0.12, 0.35, 0.15, 0.850, 0.20),
            35.0: (0.480, 0.12, 0.35, 0.15, 0.820, 0.20),
            40.0: (0.420, 0.12, 0.30, 0.15, 0.780, 0.20),
            45.0: (0.350, 0.11, 0.23, 0.15, 0.660, 0.18),
            50.0: (0.275, 0.09, 0.18, 0.15, 0.540, 0.18),
            55.0: (0.253, 0.09, 0.15, 0.15, 0.480, 0.17),
            60.0: (0.209, 0.09, 0.10, 0.15, 0.456, 0.17),
            65.0: (0.198, 0.08, 0.10, 0.15, 0.432, 0.17),
            70.0: (0.209, 0.08, 0.10, 0.15, 0.420, 0.16),
            75.0: (0.209, 0.08, 0.10, 0.15, 0.420, 0.16),
            80.0: (0.209, 0.08, 0.10, 0.15, 0.420, 0.16),
        }
        # Dict tuple order: (wall_accept_r, wall_kappa_oneshot, foot_track_kappa,
        # base_orient_kappa, landing_accept_r, landing_kappa_oneshot)

        wococo_foot_track_kappa          = 0.20
        wococo_foot_target_kappa_oneshot = 0.12
        wococo_landing_foot_target_kappa_oneshot = 0.20
        wococo_foot_target_kappa         = 0.20
        wococo_foot_target_force_thresh  = 20.0

        enable_premature_wall_termination = False

        # Plant-FSM eases reverted to 125a41f: full-strength stage-0 rear-air
        # gate and 40 deg stage-1 yaw tolerance from level 0 (no bootstrap ease).
        enable_wococo_plant_fail_termination = True
        wococo_plant_fail_term_arm_mean_level = 4.0
        # Kill the episode when the base overshoots the landing footholds toward spawn
        # by more than this. RAISED 0.15 -> 0.45 for the posture-only landing, putting
        # it in DIRECT CONFLICT with the stage-2 gate, which demands
        # wococo_landing_speed_frac * cmd for
        # wococo_landing_n_persist steps, i.e. the base MUST travel
        #     0.8 * 1.0 m/s * 15 * 0.02 s = 0.24 m
        # toward spawn while holding it. At margin 0.15 the robot died after 0.15 m,
        # before the required hold could finish: measured at level 9, 57% of episodes
        # (12/21) died on wococo_stage2_overshoot.
        # 0.45 = 0.24 mandatory travel + 0.21 slack. KEEP THIS >= that product
        # if n_persist / speed_frac / the command band change.
        wococo_stage2_overshoot_margin_m     = 0.45
        wococo_stage1_ground_max_z           = 0.05

        enable_wococo_armed_term_exempt      = True
        wococo_armed_term_exempt_min_mean_level = 6.0
        wococo_armed_term_exempt_ramp_steps  = 0
        wococo_armed_roll_term_max_deg       = None
        wococo_armed_pitch_term_max_deg      = None
        wococo_armed_disable_attitude_terms_at_full = True
        wococo_armed_base_contact_grace_steps = None

        # Anti-exploit: cap on positive sub-sentinel shaping income (contact_pts /
        # foot_track / base_orient) per episode. Raised 3.0 -> 3.1 for the posture-only
        # landing, whose dense shaping rides on the capped base_orient while the wall
        # stages exhaust 3.0 alone. The landing branch earns at most
        # scale * gain * dt * 1.0 * n_persist = 0.25 * 1.5 * 0.02 * 1.0 * 15 = 0.1125,
        # so +0.15 fits it. Re-derive if the gain or n_persist changes.
        wococo_income_episode_cap = 3.15
        enable_plant_dwell_termination = False
        wococo_plant_dwell_max_steps   = 200
        wococo_n_persist = 5   # anti-tap: hold the foothold N steps to bank
        wococo_stage0_n_persist = 5
        wococo_stage1_n_persist = 3
        # Landing hold, set to the reported value of 1.
        # WARNING: under wococo_landing_posture_only there is no foot-contact or
        # foothold test, so this hold is the ONLY thing preventing the stage from
        # completing in mid-air. A ballistic body entering [h_lo, h_hi] from above
        # satisfies the height band for sqrt(2*(h_hi-h_lo)/g) = 0.202 s = 10.1
        # steps, and horizontal speed / attitude are all satisfiable in flight.
        # At 1 the landing was measured completing in a SINGLE mid-air step; 15
        # steps (0.30 s) cleared the ballistic window by ~48%. Persistence 1 is only
        # sound together with the foothold-based landing (wococo_landing_posture_only
        # = False plus the >5 N in-target contact test), as the reported config used it.
        wococo_landing_n_persist = 1
        wococo_foot_region_radius         = 0.30
        # [swimmer] Per-level wall-press force gate. legged_robot.py prefers this table
        # when it is a non-empty 1-D sequence and falls back to the scalar below only
        # when it is None, so the scalar is inert as written.
        wococo_wall_press_force_threshold_by_level = [10.0, 10.0, 35.0, 45.0, 50.0]
        wococo_wall_press_force_threshold = 10.0   # 125a41f flat gate (fallback only)
        wococo_arm_x_offset_m             = 0.0
        # Jackpot discount min(run_len,K)/K; run_len caps at the stage n_persist, so K=1 at the landing restores the full one-shot bonus.
        wococo_jackpot_persist_K = 6.0
        wococo_landing_jackpot_persist_K = 1.0
        hip_pos_episode_multiplier = 1.0

        class scales( Go2RoughCfg.rewards.scales ):
            # ---- IsaacLab UnitreeGo2RoughEnvCfg.RewardsCfg, all eleven terms ----
            # Base weights: velocity_env_cfg.py:222-255. Go2 overrides:
            # config/go2/rough_env_cfg.py:50-57. Names follow this repo's
            # _reward_* methods, with the IsaacLab term in the comment.
            #
            # Nine of the eleven already existed here under another name, so
            # they are re-weighted in place. Listing both spellings would put
            # two active entries on one physical quantity and double the
            # penalty, which is why only the two genuinely new terms
            # (feet_air_time, undesired_contacts) are additions.
            tracking_goal_vel  = 1.5     # track_lin_vel_xy_exp  (this repo: min/div kernel)
            tracking_yaw       = 0.75    # track_ang_vel_z_exp   (was 0.5)
            lin_vel_z          = -2.0    # lin_vel_z_l2          (was -1.0)
            ang_vel_xy         = -0.05   # ang_vel_xy_l2
            torques            = -0.0002 # dof_torques_l2        (was -1e-5, Go2 tightens it 20x)
            dof_acc            = -2.5e-7 # dof_acc_l2
            action_rate        = -0.01   # action_rate_l2        (was -0.1)
            feet_air_time      = 0.01    # feet_air_time         (NEW; Go2 lowers 0.125 to 0.01)
            undesired_contacts = 0.0     # undesired_contacts    (NEW; IsaacLab Go2 sets it None)
            # IsaacLab leaves flat_orientation_l2 and dof_pos_limits at 0.0 for
            # Go2. flat_orientation_l2 is this repo's `orientation`, which keeps
            # its flat-terrain gate; dof_pos_limits stays tight on purpose.
            orientation        = -1.0
            dof_pos_limits     = -10.0

            # ---- Terms with no IsaacLab counterpart ----
            collision = -20.
            standstill_orientation = 0.5
            feet_contact_forces = -0.01
            termination = -150.0
            timeout     = 0.0

            # The WoCoCo / hurdle / wall set is gone along with the task. Those
            # rewards are gated on env_class 15 or 19 (_reward_wococo_stage,
            # _reward_wococo_contact_pts), and terrain_dict leaves every wall
            # slot at 0.0, so env_class never reaches either value: all of them
            # returned 0 on every step while still costing a function call and
            # a logged curve apiece. The config knobs they read still exist on
            # Go2TeacherCfg.rewards with their defaults, so nothing else broke.


class Go2TeacherCfgPPO( Go2RoughCfgPPO ):
    class policy( Go2RoughCfgPPO.policy ):
        init_noise_std = 1.0
        max_noise_std  = 1.0
