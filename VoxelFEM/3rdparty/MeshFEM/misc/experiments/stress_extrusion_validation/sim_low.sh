Simulate_cli pattern_extrude_0.06125_thick.msh    -b pull_y_dirichet_3d.bc -m /home/fjp234/microstructures/materials/B9Creator.material -o $SCRATCH/extruded_sims/sim_0.06125.msh     > $SCRATCH/extruded_sims/sim_0.06125.txt
Simulate_cli pattern_extrude_0.06125_thick_x2.msh -b pull_y_dirichet_3d.bc -m /home/fjp234/microstructures/materials/B9Creator.material -o $SCRATCH/extruded_sims/sim_0.06125_x2.msh  > $SCRATCH/extruded_sims/sim_0.06125_x2.txt
Simulate_cli pattern_extrude_0.125_thick.msh    -b pull_y_dirichet_3d.bc -m /home/fjp234/microstructures/materials/B9Creator.material -o $SCRATCH/extruded_sims/sim_0.125.msh     > $SCRATCH/extruded_sims/sim_0.125.txt
Simulate_cli pattern_extrude_0.125_thick_x2.msh -b pull_y_dirichet_3d.bc -m /home/fjp234/microstructures/materials/B9Creator.material -o $SCRATCH/extruded_sims/sim_0.125_x2.msh  > $SCRATCH/extruded_sims/sim_0.125_x2.txt
