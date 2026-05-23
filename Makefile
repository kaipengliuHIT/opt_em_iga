# Makefile for SOI waveguide simulation

# MFEM build configuration (use cuda build for GPU support)
MFEM_BUILD_DIR = /mnt/d/code/code_opt_em/mfem-cuda-build
MFEM_SRC_DIR = /mnt/d/code/code_opt_em/mfem
include $(MFEM_BUILD_DIR)/config/config.mk

OPT_DIR = /mnt/d/code/code_opt_em/opt
MFEM_TPLFLAGS = -I$(OPT_DIR)/hypre/include -I$(OPT_DIR)/metis/include -I$(OPT_DIR)/openmpi/include
MFEM_EXT_LIBS = -Wl,-rpath,$(OPT_DIR)/hypre/lib -L$(OPT_DIR)/hypre/lib -lHYPRE \
                $(OPT_DIR)/metis/lib/libmetis.a \
                -Wl,-rpath,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -lcusparse \
                -Wl,-rpath,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -lcublas \
                -Wl,-rpath,$(OPT_DIR)/openmpi/lib -L$(OPT_DIR)/openmpi/lib -lmpi

# CUDA and cuDSS paths
CUDA_DIR = /usr/local/cuda-12.8
CUDSS_LIB_DIR = /usr/lib/x86_64-linux-gnu/libcudss/12

# Compiler and flags
CXX = $(MFEM_CXX)
CXXFLAGS = $(MFEM_CXXFLAGS) -I$(MFEM_SRC_DIR) -I$(MFEM_BUILD_DIR) -I$(MFEM_BUILD_DIR)/config -I./to_nurbs_patch -I$(CUDA_DIR)/include -I$(CUDA_DIR)/targets/x86_64-linux/include $(MFEM_TPLFLAGS)
DEBUG_CXXFLAGS = -std=c++17 -O0 -g -I$(MFEM_SRC_DIR) -I$(MFEM_BUILD_DIR) -I$(MFEM_BUILD_DIR)/config -I./to_nurbs_patch -I$(CUDA_DIR)/include -I$(CUDA_DIR)/targets/x86_64-linux/include $(MFEM_TPLFLAGS)

# Linker flags (directly link to CUDA 12 version of cuDSS)
LDFLAGS = -L$(MFEM_BUILD_DIR) -lmfem $(MFEM_EXT_LIBS) -L$(CUDA_DIR)/lib64 -lcudart -lcublas -lcublasLt -Wl,-rpath,$(CUDSS_LIB_DIR) $(CUDSS_LIB_DIR)/libcudss.so.0.7.1
LDFLAGS_NO_CUDSS = -L$(MFEM_BUILD_DIR) -lmfem $(MFEM_EXT_LIBS) -L$(CUDA_DIR)/lib64 -lcudart -lcublas -lcublasLt \
                   -Wl,-rpath,$(OPT_DIR)/openmpi/lib -Wl,-rpath,$(OPT_DIR)/hypre/lib

# Target executables
TARGET = soi
MODE_SOLVER = mode_solver

# Source files
SRCS = soi.cpp to_nurbs_patch/pixel_to_nurbs.cpp waveguide_mode_solver.cpp analytical_mode_solver.cpp
OBJS = $(SRCS:.cpp=.o)

# Default target
all: $(TARGET) $(MODE_SOLVER)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Mode solver standalone
$(MODE_SOLVER): mode_solver_standalone.o waveguide_mode_solver.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

mode_solver_standalone.o: mode_solver_standalone.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Analytical mode solver (C++)
analytical_solver: analytical_mode_main.o analytical_mode_solver.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

analytical_mode_solver.o: analytical_mode_solver.cpp analytical_mode_solver.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

analytical_mode_main.o: analytical_mode_main.cpp analytical_mode_solver.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Uniform medium test
soi_uniform_test: soi_uniform_test.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

soi_uniform_test.o: soi_uniform_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# QAAQ/TFSF unidirectional source
soi_qaaq: soi_qaaq.o to_nurbs_patch/pixel_to_nurbs.o waveguide_mode_solver.o analytical_mode_solver.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

soi_qaaq.o: soi_qaaq.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Mode field computation (Step 1 of QAAQ)
soi_mode_field: soi_mode_field.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

soi_mode_field.o: soi_mode_field.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 90-degree bend topology optimization
bend_topology_opt: bend_topology_opt.o mma_optimizer.o analytical_mode_solver.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

bend_topology_opt.o: bend_topology_opt.cpp mma_optimizer.hpp analytical_mode_solver.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

mma_optimizer.o: mma_optimizer.cpp mma_optimizer.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# SOI topology optimization (based on soi_qaaq)
soi_topology_opt: soi_topology_opt.o mma_optimizer.o analytical_mode_solver.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

soi_topology_opt.o: soi_topology_opt.cpp mma_optimizer.hpp analytical_mode_solver.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 2D Focusing - Adjoint sensitivity verification
focusing_2d: focusing_2d.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

focusing_2d.o: focusing_2d.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Poisson adjoint test
adjoint_test_poisson: adjoint_test_poisson.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

adjoint_test_poisson.o: adjoint_test_poisson.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Helmholtz adjoint test
adjoint_test_helmholtz: adjoint_test_helmholtz.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

adjoint_test_helmholtz.o: adjoint_test_helmholtz.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 2D Focusing optimization
focusing_2d_opt: focusing_2d_opt.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

focusing_2d_opt.o: focusing_2d_opt.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -f $(TARGET) $(MODE_SOLVER) analytical_solver soi_uniform_test bend_topology_opt $(OBJS) mode_solver_standalone.o analytical_mode_solver.o analytical_mode_main.o mma_optimizer.o bend_topology_opt.o

# Run with MPI (4 processes)
run: $(TARGET)
	$(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) 4 ./$(TARGET) -m $(MFEM_SRC_DIR)/data/cube-nurbs.mesh

# Run with 1 process for testing
run1: $(TARGET)
	$(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) 1 ./$(TARGET) -m $(MFEM_SRC_DIR)/data/cube-nurbs.mesh

# Run 90-degree bend optimization
run_bend: bend_topology_opt
	$(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) 4 ./bend_topology_opt -m $(MFEM_DIR)/data/cube-nurbs.mesh -r 2 -mi 50

run_bend1: bend_topology_opt
	mpirun -np 1 ./bend_topology_opt -m $(MFEM_DIR)/data/cube-nurbs.mesh -r 2 -mi 20

# Run SOI topology optimization
run_topo: soi_topology_opt
	mpirun -np 4 ./soi_topology_opt -m $(MFEM_SRC_DIR)/data/cube-nurbs.mesh -r 3 -mi 30 -nx 15 -ny 15

.PHONY: all clean run run1 run_bend run_bend1 run_topo

focusing_lens_opt: focusing_lens_opt.o
	$(CXX) $(CXXFLAGS) -o focusing_lens_opt focusing_lens_opt.o $(LDFLAGS)

focusing_lens_opt.o: focusing_lens_opt.cpp
	$(CXX) $(CXXFLAGS) -c focusing_lens_opt.cpp -o focusing_lens_opt.o

focusing_lens_opt2: focusing_lens_opt2.o
	$(CXX) $(CXXFLAGS) -o focusing_lens_opt2 focusing_lens_opt2.o $(LDFLAGS)

focusing_lens_opt2.o: focusing_lens_opt2.cpp
	$(CXX) $(CXXFLAGS) -c focusing_lens_opt2.cpp -o focusing_lens_opt2.o

lens_opt_v3: lens_opt_v3.o
	$(CXX) $(CXXFLAGS) -o lens_opt_v3 lens_opt_v3.o $(LDFLAGS)

lens_opt_v3.o: lens_opt_v3.cpp
	$(CXX) $(CXXFLAGS) -c lens_opt_v3.cpp -o lens_opt_v3.o

# GPU version
lens_opt_gpu: lens_opt_gpu.o
	$(CXX) $(CXXFLAGS) -o lens_opt_gpu lens_opt_gpu.o $(LDFLAGS)

lens_opt_gpu.o: lens_opt_gpu.cpp
	$(CXX) $(CXXFLAGS) -c lens_opt_gpu.cpp -o lens_opt_gpu.o

lens_opt_gpu_v2: lens_opt_gpu_v2.o
	$(CXX) $(CXXFLAGS) -o lens_opt_gpu_v2 lens_opt_gpu_v2.o $(LDFLAGS)

lens_opt_gpu_v2.o: lens_opt_gpu_v2.cpp
	$(CXX) $(CXXFLAGS) -c lens_opt_gpu_v2.cpp -o lens_opt_gpu_v2.o

lens_opt_gpu_v3: lens_opt_gpu_v3.o
	$(CXX) $(CXXFLAGS) -o lens_opt_gpu_v3 lens_opt_gpu_v3.o $(LDFLAGS)

lens_opt_gpu_v3.o: lens_opt_gpu_v3.cpp
	$(CXX) $(CXXFLAGS) -c lens_opt_gpu_v3.cpp -o lens_opt_gpu_v3.o

# 2D Waveguide Bend
waveguide_bend_2d: waveguide_bend_2d.o
	$(CXX) $(CXXFLAGS) -o waveguide_bend_2d waveguide_bend_2d.o $(LDFLAGS)

waveguide_bend_2d.o: waveguide_bend_2d.cpp
	$(CXX) $(CXXFLAGS) -c waveguide_bend_2d.cpp -o waveguide_bend_2d.o

# 3D Waveguide Bend (SOI, H(curl), AMS)
waveguide_bend_3d: waveguide_bend_3d.o analytical_mode_solver.o
	$(CXX) $(CXXFLAGS) -o waveguide_bend_3d waveguide_bend_3d.o analytical_mode_solver.o $(LDFLAGS)

waveguide_bend_3d.o: waveguide_bend_3d.cpp analytical_mode_solver.hpp
	$(CXX) $(CXXFLAGS) -c waveguide_bend_3d.cpp -o waveguide_bend_3d.o

analytical_mode_solver.o: analytical_mode_solver.cpp analytical_mode_solver.hpp
	$(CXX) $(CXXFLAGS) -c analytical_mode_solver.cpp -o analytical_mode_solver.o

# Adjoint gradient test (fully modular version with cuDSS support)
adjoint_gradient_test: em_opt/adjoint_gradient_test.o em_opt/cudss_solver_real.o em_opt/cudss_mpi_solver.o analytical_mode_solver.o
	$(CXX) $(CXXFLAGS) -o adjoint_gradient_test em_opt/adjoint_gradient_test.o em_opt/cudss_solver_real.o em_opt/cudss_mpi_solver.o analytical_mode_solver.o $(LDFLAGS)

em_opt/adjoint_gradient_test.o: em_opt/adjoint_gradient_test.cpp analytical_mode_solver.hpp em_opt/pml_coefficients.hpp em_opt/qaaq_source.hpp em_opt/mode_overlap.hpp em_opt/adjoint_solver.hpp em_opt/gradient_calculator.hpp em_opt/cudss_solver_real.hpp em_opt/cudss_mpi_solver.hpp
	$(CXX) $(CXXFLAGS) -I./em_opt -c em_opt/adjoint_gradient_test.cpp -o em_opt/adjoint_gradient_test.o


# Adjoint optimization loop (reuses verified modules + cuDSS)
adjoint_opt_loop: em_opt/adjoint_opt_loop.o em_opt/cudss_solver_real.o em_opt/cudss_mpi_solver.o analytical_mode_solver.o
	$(CXX) $(CXXFLAGS) -o adjoint_opt_loop em_opt/adjoint_opt_loop.o em_opt/cudss_solver_real.o em_opt/cudss_mpi_solver.o analytical_mode_solver.o $(LDFLAGS)

em_opt/adjoint_opt_loop.o: em_opt/adjoint_opt_loop.cpp analytical_mode_solver.hpp em_opt/pml_coefficients.hpp em_opt/qaaq_source.hpp em_opt/mode_overlap.hpp em_opt/adjoint_solver.hpp em_opt/gradient_calculator.hpp em_opt/cudss_solver_real.hpp em_opt/cudss_mpi_solver.hpp
	$(CXX) $(CXXFLAGS) -I./em_opt -c em_opt/adjoint_opt_loop.cpp -o em_opt/adjoint_opt_loop.o

em_opt/cudss_solver_real.o: em_opt/cudss_solver_real.cpp em_opt/cudss_solver_real.hpp
	$(CXX) $(CXXFLAGS) -I./em_opt -c em_opt/cudss_solver_real.cpp -o em_opt/cudss_solver_real.o

em_opt/cudss_mpi_solver.o: em_opt/cudss_mpi_solver.cpp em_opt/cudss_mpi_solver.hpp
	$(CXX) $(CXXFLAGS) -I./em_opt -c em_opt/cudss_mpi_solver.cpp -o em_opt/cudss_mpi_solver.o

# Topology optimization example (modular version)
topology_opt_example: em_opt/topology_optimization_example.o analytical_mode_solver.o
	$(CXX) $(CXXFLAGS) -o topology_opt_example em_opt/topology_optimization_example.o analytical_mode_solver.o $(LDFLAGS)

# Single-patch reference-space FDFD initializer prototype
fdfd_iga_init_demo: \
	fdfd_iga_init/single_patch_demo.o \
	fdfd_iga_init/reference_fdfd_cpu.o \
	fdfd_iga_init/reference_fdfd_cuda.o \
	fdfd_iga_init/reference_initial_guess.o \
	covariant_aux_space/covariant_aux_preconditioner.o \
	covariant_aux_space/yee_transfer.o \
	covariant_aux_space/yee_operator.o \
	covariant_aux_space/covariant_reference_preconditioner.o
	$(CXX) $(CXXFLAGS) -I./fdfd_iga_init -o fdfd_iga_init_demo $^ $(LDFLAGS_NO_CUDSS)

fdfd_iga_init/single_patch_demo.o: \
	fdfd_iga_init/single_patch_demo.cpp \
	fdfd_iga_init/reference_fdfd_cpu.hpp \
	fdfd_iga_init/reference_initial_guess.hpp
	$(CXX) $(CXXFLAGS) -I./fdfd_iga_init -c fdfd_iga_init/single_patch_demo.cpp -o $@

fdfd_iga_init/reference_fdfd_cpu.o: \
	fdfd_iga_init/reference_fdfd_cpu.cpp \
	fdfd_iga_init/reference_fdfd_cpu.hpp \
	fdfd_iga_init/reference_patch_evaluator.hpp
	$(CXX) $(CXXFLAGS) -I./fdfd_iga_init -c fdfd_iga_init/reference_fdfd_cpu.cpp -o $@

fdfd_iga_init/reference_fdfd_cuda.o: \
	fdfd_iga_init/reference_fdfd_cuda.cpp \
	fdfd_iga_init/reference_fdfd_cuda.hpp \
	fdfd_iga_init/reference_fdfd_cpu.hpp
	$(CXX) $(CXXFLAGS) -I./fdfd_iga_init -c fdfd_iga_init/reference_fdfd_cuda.cpp -o $@

fdfd_iga_init/reference_initial_guess.o: \
	fdfd_iga_init/reference_initial_guess.cpp \
	fdfd_iga_init/reference_initial_guess.hpp \
	fdfd_iga_init/reference_fdfd_cpu.hpp
	$(CXX) $(CXXFLAGS) -I./fdfd_iga_init -c fdfd_iga_init/reference_initial_guess.cpp -o $@

covariant_aux_space/covariant_aux_preconditioner.o: \
	covariant_aux_space/covariant_aux_preconditioner.cpp \
	covariant_aux_space/covariant_aux_preconditioner.hpp
	$(CXX) $(CXXFLAGS) -I./covariant_aux_space -c covariant_aux_space/covariant_aux_preconditioner.cpp -o $@

covariant_aux_space/yee_transfer.o: \
	covariant_aux_space/yee_transfer.cpp \
	covariant_aux_space/yee_transfer.hpp \
	fdfd_iga_init/reference_fdfd_cpu.hpp
	$(CXX) $(CXXFLAGS) -I./covariant_aux_space -I./fdfd_iga_init -c covariant_aux_space/yee_transfer.cpp -o $@

covariant_aux_space/yee_operator.o: \
	covariant_aux_space/yee_operator.cpp \
	covariant_aux_space/yee_operator.hpp \
	fdfd_iga_init/reference_fdfd_cpu.hpp \
	fdfd_iga_init/reference_patch_evaluator.hpp
	$(CXX) $(CXXFLAGS) -I./covariant_aux_space -I./fdfd_iga_init -c covariant_aux_space/yee_operator.cpp -o $@

covariant_aux_space/covariant_reference_preconditioner.o: \
	covariant_aux_space/covariant_reference_preconditioner.cpp \
	covariant_aux_space/covariant_reference_preconditioner.hpp \
	covariant_aux_space/yee_operator.hpp \
	covariant_aux_space/yee_transfer.hpp \
	covariant_aux_space/covariant_aux_preconditioner.hpp \
	fdfd_iga_init/reference_fdfd_cpu.hpp \
	fdfd_iga_init/reference_initial_guess.hpp
	$(CXX) $(CXXFLAGS) -I./covariant_aux_space -I./fdfd_iga_init -c covariant_aux_space/covariant_reference_preconditioner.cpp -o $@

run_fdfd_iga_init_demo: fdfd_iga_init_demo
	LD_LIBRARY_PATH=$(OPT_DIR)/openmpi/lib:$(OPT_DIR)/hypre/lib:$$LD_LIBRARY_PATH ./fdfd_iga_init_demo

scan_fdfd_iga_init_demo: fdfd_iga_init_demo
	LD_LIBRARY_PATH=$(OPT_DIR)/openmpi/lib:$(OPT_DIR)/hypre/lib:$$LD_LIBRARY_PATH ./fdfd_iga_init_demo --scan --scan-orders 2,3 --scan-refines 1,2 --scan-fd 31,41 --scan-aux 5,7 --scan-max-ratio 0.8

rebuild_fdfd_iga_init_demo:
	rm -f fdfd_iga_init_demo \
	      fdfd_iga_init/single_patch_demo.o \
	      fdfd_iga_init/reference_fdfd_cpu.o \
	      fdfd_iga_init/reference_fdfd_cuda.o \
	      fdfd_iga_init/reference_initial_guess.o \
	      covariant_aux_space/covariant_aux_preconditioner.o \
	      covariant_aux_space/yee_transfer.o \
	      covariant_aux_space/yee_operator.o \
	      covariant_aux_space/covariant_reference_preconditioner.o
	$(MAKE) fdfd_iga_init_demo

gdb_fdfd_iga_init_demo: fdfd_iga_init_demo
	LD_LIBRARY_PATH=$(OPT_DIR)/openmpi/lib:$(OPT_DIR)/hypre/lib:$$LD_LIBRARY_PATH gdb -q ./fdfd_iga_init_demo -ex run -ex bt -ex quit

fdfd_iga_init_demo_debug: \
	fdfd_iga_init/single_patch_demo.debug.o \
	fdfd_iga_init/reference_fdfd_cpu.debug.o \
	fdfd_iga_init/reference_fdfd_cuda.debug.o \
	fdfd_iga_init/reference_initial_guess.debug.o \
	covariant_aux_space/covariant_aux_preconditioner.debug.o \
	covariant_aux_space/yee_transfer.debug.o \
	covariant_aux_space/yee_operator.debug.o \
	covariant_aux_space/covariant_reference_preconditioner.debug.o
	$(CXX) $(DEBUG_CXXFLAGS) -I./fdfd_iga_init -o fdfd_iga_init_demo_debug $^ $(LDFLAGS_NO_CUDSS)

fdfd_iga_init/single_patch_demo.debug.o: \
	fdfd_iga_init/single_patch_demo.cpp \
	fdfd_iga_init/reference_fdfd_cpu.hpp \
	fdfd_iga_init/reference_initial_guess.hpp
	$(CXX) $(DEBUG_CXXFLAGS) -I./fdfd_iga_init -c fdfd_iga_init/single_patch_demo.cpp -o $@

fdfd_iga_init/reference_fdfd_cpu.debug.o: \
	fdfd_iga_init/reference_fdfd_cpu.cpp \
	fdfd_iga_init/reference_fdfd_cpu.hpp \
	fdfd_iga_init/reference_patch_evaluator.hpp
	$(CXX) $(DEBUG_CXXFLAGS) -I./fdfd_iga_init -c fdfd_iga_init/reference_fdfd_cpu.cpp -o $@

fdfd_iga_init/reference_fdfd_cuda.debug.o: \
	fdfd_iga_init/reference_fdfd_cuda.cpp \
	fdfd_iga_init/reference_fdfd_cuda.hpp \
	fdfd_iga_init/reference_fdfd_cpu.hpp
	$(CXX) $(DEBUG_CXXFLAGS) -I./fdfd_iga_init -c fdfd_iga_init/reference_fdfd_cuda.cpp -o $@

fdfd_iga_init/reference_initial_guess.debug.o: \
	fdfd_iga_init/reference_initial_guess.cpp \
	fdfd_iga_init/reference_initial_guess.hpp \
	fdfd_iga_init/reference_fdfd_cpu.hpp
	$(CXX) $(DEBUG_CXXFLAGS) -I./fdfd_iga_init -c fdfd_iga_init/reference_initial_guess.cpp -o $@

covariant_aux_space/covariant_aux_preconditioner.debug.o: \
	covariant_aux_space/covariant_aux_preconditioner.cpp \
	covariant_aux_space/covariant_aux_preconditioner.hpp
	$(CXX) $(DEBUG_CXXFLAGS) -I./covariant_aux_space -c covariant_aux_space/covariant_aux_preconditioner.cpp -o $@

covariant_aux_space/yee_transfer.debug.o: \
	covariant_aux_space/yee_transfer.cpp \
	covariant_aux_space/yee_transfer.hpp \
	fdfd_iga_init/reference_fdfd_cpu.hpp
	$(CXX) $(DEBUG_CXXFLAGS) -I./covariant_aux_space -I./fdfd_iga_init -c covariant_aux_space/yee_transfer.cpp -o $@

covariant_aux_space/yee_operator.debug.o: \
	covariant_aux_space/yee_operator.cpp \
	covariant_aux_space/yee_operator.hpp \
	fdfd_iga_init/reference_fdfd_cpu.hpp \
	fdfd_iga_init/reference_patch_evaluator.hpp
	$(CXX) $(DEBUG_CXXFLAGS) -I./covariant_aux_space -I./fdfd_iga_init -c covariant_aux_space/yee_operator.cpp -o $@

covariant_aux_space/covariant_reference_preconditioner.debug.o: \
	covariant_aux_space/covariant_reference_preconditioner.cpp \
	covariant_aux_space/covariant_reference_preconditioner.hpp \
	covariant_aux_space/yee_operator.hpp \
	covariant_aux_space/yee_transfer.hpp \
	covariant_aux_space/covariant_aux_preconditioner.hpp \
	fdfd_iga_init/reference_fdfd_cpu.hpp \
	fdfd_iga_init/reference_initial_guess.hpp
	$(CXX) $(DEBUG_CXXFLAGS) -I./covariant_aux_space -I./fdfd_iga_init -c covariant_aux_space/covariant_reference_preconditioner.cpp -o $@

gdb_fdfd_iga_init_demo_debug: fdfd_iga_init_demo_debug
	LD_LIBRARY_PATH=$(OPT_DIR)/openmpi/lib:$(OPT_DIR)/hypre/lib:$$LD_LIBRARY_PATH gdb -q ./fdfd_iga_init_demo_debug -ex run -ex bt -ex quit

em_opt/topology_optimization_example.o: em_opt/topology_optimization_example.cpp analytical_mode_solver.hpp
	$(CXX) $(CXXFLAGS) -I./em_opt -c em_opt/topology_optimization_example.cpp -o em_opt/topology_optimization_example.o

# Point source test
test_point_source: test_point_source.o em_opt/cudss_mpi_solver.o
	$(CXX) $(CXXFLAGS) -o test_point_source test_point_source.o em_opt/cudss_mpi_solver.o $(LDFLAGS)

test_point_source.o: test_point_source.cpp em_opt/pml_coefficients.hpp em_opt/cudss_mpi_solver.hpp
	$(CXX) $(CXXFLAGS) -I./em_opt -c test_point_source.cpp -o test_point_source.o
