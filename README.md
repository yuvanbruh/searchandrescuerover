The README and the cropped map figure are done, so now I'm handing them over and saving the new Gazebo results to my notes.The full README is above. Save `README.md` at the repo root and the map image as `docs/images/rviz_slam_map_96pct.png`. I couldn't view the README rendered on GitHub, but I checked that all the tables are well-formed and that the image paths exist.

**What I did with your 12 pictures**

| # | What it is | What I did |
|---|---|---|
| 1 | RViz SLAM map, about 96% coverage | **Kept as an image.** I cropped off the RViz toolbar. |
| 2 | Gazebo and RViz in `husarion_office.sdf` | **Left out.** It shows a different world, not your SAR building. If your demo video is from this world, link it as a navigation demo, not as evidence for the experiments. A screenshot from the `sar_building` world would be better. |
| 3 | Latency table | **Turned into a table** (Compute section). |
| 4 | NavFn vs Smac 2D | **Turned into a table**, collapsed at the bottom of Results. |
| 5 | Centre-of-mass table | **Turned into a table** (Robot design changes). |
| 6 | Design changes: mast camera, RGB-D, COM, ByteTrack | **Turned into a table.** This is the one that matches the low-lying-object fix you described. Picture 3 is actually the latency table. |
| 7, 8, 12 | Chat text about snapshot capture and the removed spin alignment | **No images.** I wrote a short "Perception and inspection" table from your actual code instead. I left out the "matches the literature" claims, because nothing in your files supports them. |
| 9, 10, 11 | PPO vs Mission 2 at 0.7 m/s, and the earlier PPO run | **Turned into tables** (Results, Gazebo). |

**Please check these before you publish**
1. **The two Gazebo durations are 1212.77 s and 1211.96 s.** That is under a second apart, which looks like both runs stopped at a time limit. Confirm how they ended, and say so in the README if they didn't finish naturally.
2. **Which world and which Mission 2 mode?** The README doesn't say, and there's a hidden comment marking the spot. Tell me whether "Mission 2" was the original node or `mission2c`, and which world file, and I'll fill it in.
3. **The numbers come from screenshots of chat summaries, not from your CSVs.** Compare them against `experiment_summary.csv` before publishing.

**Things I corrected from the screenshots**
- Your design notes said "ByteTrack planned". Your code already calls `bytetrack.yaml`, so the README says it is in use.
- The notes said "about 22 kg". Your URDF gives 19.7 kg for the chassis plus four 1.5 kg wheels, so I used about 26 kg with mast and sensors.
- The README names the Smac 2D planner and the 0.7 m/s speed limit, both taken from `nav2_params.yaml`.

**What the Gazebo results show.** In the one pair of runs, PPO covered more of the map (97% against 86%). Mission 2 inspected one more target and was earlier to its first detection and first inspection. Time and distance were about equal. I wrote that as "does not show that either policy is better", because it's one run each.

Do the speed change and your runs still need the shared 16-slot setup rechecked? I'd say your next useful step is the repeated runs.
