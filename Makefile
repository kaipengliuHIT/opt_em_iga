# Makefile for SOI waveguide simulation

# MFEM build configuration. Override these on the make command line if needed.
WORKSPACE_DIR ?= /mnt/f/optemcode
MFEM_BUILD_DIR ?= $(WORKSPACE_DIR)/mfem-cpu-build
MFEM_SRC_DIR ?= $(WORKSPACE_DIR)/mfem
include $(MFEM_BUILD_DIR)/config/config.mk

OPT_DIR ?= $(WORKSPACE_DIR)/opt
MFEM_TPLFLAGS = -I$(OPT_DIR)/hypre/include -I$(OPT_DIR)/metis/include -I$(OPT_DIR)/openmpi/include
MFEM_EXT_LIBS = -Wl,-rpath,$(OPT_DIR)/hypre/lib -L$(OPT_DIR)/hypre/lib -lHYPRE \
                $(OPT_DIR)/metis/lib/libmetis.a \
                -Wl,-rpath,$(OPT_DIR)/openmpi/lib -L$(OPT_DIR)/openmpi/lib -lmpi

# CUDA 12.8 system toolkit (installed via: sudo apt install cuda-toolkit-12-8)
CUDA_DIR          = /usr/local/cuda-12.8
CUDSS_LIB_DIR     = /usr/lib/x86_64-linux-gnu/libcudss/12

CUDA_INCLUDE_DIRS = -I$(CUDA_DIR)/include
CUDA_LIB_DIRS     = $(CUDA_DIR)/lib64
CUDA_RPATH        = -Wl,-rpath,$(CUDA_DIR)/lib64
CUDA_LINK         = -L$(CUDA_DIR)/lib64 $(CUDA_RPATH) -lcublas -lcublasLt -lcudart

# Compiler and flags
CXX = $(MFEM_CXX)
CXXFLAGS = $(MFEM_CXXFLAGS) -I$(MFEM_SRC_DIR) -I$(MFEM_BUILD_DIR) -I$(MFEM_BUILD_DIR)/config -I./to_nurbs_patch $(MFEM_TPLFLAGS)
DEBUG_CXXFLAGS = -std=c++17 -O0 -g -I$(MFEM_SRC_DIR) -I$(MFEM_BUILD_DIR) -I$(MFEM_BUILD_DIR)/config -I./to_nurbs_patch $(MFEM_TPLFLAGS)

# Linker flags (directly link to CUDA 12 version of cuDSS)
LDFLAGS = -L$(MFEM_BUILD_DIR) -lmfem $(MFEM_EXT_LIBS) -L$(CUDA_DIR)/lib64 -lcudart -lcublas -lcublasLt -Wl,-rpath,$(CUDSS_LIB_DIR) $(CUDSS_LIB_DIR)/libcudss.so.0.7.1
LDFLAGS_NO_CUDSS = -L$(MFEM_BUILD_DIR) -lmfem $(MFEM_EXT_LIBS) \
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

# ── GPU BiCGSTAB solver (maxwellb-style, C++) ─────────────────────────────────
CUDA_SOLVER_DIR = cuda_iterative_solver
COVARIANT_DIR   = covariant_aux_space

# Shared objects for both CPU and GPU builds
BICGSTAB_SHARED_OBJS = \
	$(CUDA_SOLVER_DIR)/gpu_bicgstab.o \
	$(COVARIANT_DIR)/iga_patch_ras_preconditioner.o

# CPU-only build: no CUDA headers needed, CpuBiCGSTABSolver only
gpu_bicgstab_demo_cpu: \
	$(CUDA_SOLVER_DIR)/gpu_bicgstab_demo.cpu.o \
	$(BICGSTAB_SHARED_OBJS)
	$(CXX) $(CXXFLAGS) -I./$(COVARIANT_DIR) -I./$(CUDA_SOLVER_DIR) \
	  -o $@ $^ $(LDFLAGS_NO_CUDSS)

$(CUDA_SOLVER_DIR)/gpu_bicgstab_demo.cpu.o: \
	$(CUDA_SOLVER_DIR)/gpu_bicgstab_demo.cpp \
	$(CUDA_SOLVER_DIR)/gpu_bicgstab.hpp \
	$(COVARIANT_DIR)/iga_patch_ras_preconditioner.hpp
	$(CXX) $(CXXFLAGS) -I./$(COVARIANT_DIR) -I./$(CUDA_SOLVER_DIR) \
	  -c $< -o $@

# GPU build: -DHAVE_CUDA, headers+libs from conda pytorch nvidia packages
GPU_LDFLAGS = $(LDFLAGS_NO_CUDSS) $(CUDA_LINK)

gpu_bicgstab_demo_gpu: \
	$(CUDA_SOLVER_DIR)/gpu_bicgstab_demo.gpu.o \
	$(CUDA_SOLVER_DIR)/gpu_bicgstab.gpu.o \
	$(COVARIANT_DIR)/iga_patch_ras_preconditioner.o
	$(CXX) $(CXXFLAGS) -DHAVE_CUDA \
	  -I./$(COVARIANT_DIR) -I./$(CUDA_SOLVER_DIR) $(CUDA_INCLUDE_DIRS) \
	  -o $@ $^ $(GPU_LDFLAGS)

$(CUDA_SOLVER_DIR)/gpu_bicgstab_demo.gpu.o: \
	$(CUDA_SOLVER_DIR)/gpu_bicgstab_demo.cpp \
	$(CUDA_SOLVER_DIR)/gpu_bicgstab.hpp \
	$(COVARIANT_DIR)/iga_patch_ras_preconditioner.hpp
	$(CXX) $(CXXFLAGS) -DHAVE_CUDA \
	  -I./$(COVARIANT_DIR) -I./$(CUDA_SOLVER_DIR) $(CUDA_INCLUDE_DIRS) \
	  -c $< -o $@

$(CUDA_SOLVER_DIR)/gpu_bicgstab.gpu.o: \
	$(CUDA_SOLVER_DIR)/gpu_bicgstab.cpp \
	$(CUDA_SOLVER_DIR)/gpu_bicgstab.hpp
	$(CXX) $(CXXFLAGS) -DHAVE_CUDA \
	  -I./$(COVARIANT_DIR) -I./$(CUDA_SOLVER_DIR) $(CUDA_INCLUDE_DIRS) \
	  -c $< -o $@

# Shared (CPU-only) object for gpu_bicgstab.cpp
$(CUDA_SOLVER_DIR)/gpu_bicgstab.o: \
	$(CUDA_SOLVER_DIR)/gpu_bicgstab.cpp \
	$(CUDA_SOLVER_DIR)/gpu_bicgstab.hpp
	$(CXX) $(CXXFLAGS) -I./$(COVARIANT_DIR) -I./$(CUDA_SOLVER_DIR) \
	  -c $< -o $@

.PHONY: all clean run run1 run_bend run_bend1 run_topo \
        gpu_bicgstab_demo_cpu gpu_bicgstab_demo_gpu \
        pml_point_source_demo pml_point_source_demo_gpu \
        pml_pi_prec_test \
        pml_split_demo

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

# NURBS 3D point-source PML demo for AMS vs edge-Yee comparison
PML_DEMO_SHARED_OBJS = \
	covariant_aux_space/yee_transfer.o \
	covariant_aux_space/yee_operator.o \
	covariant_aux_space/covariant_reference_preconditioner.o \
	covariant_aux_space/iga_patch_ras_preconditioner.o \
	fdfd_iga_init/reference_fdfd_cpu.o \
	fdfd_iga_init/reference_initial_guess.o

# CPU build: supports -solver gmres|bicgstab_cpu (default, no CUDA)
pml_point_source_demo: \
	pml_point_source_demo.o \
	$(CUDA_SOLVER_DIR)/gpu_bicgstab.o \
	$(PML_DEMO_SHARED_OBJS)
	$(CXX) $(CXXFLAGS) -I./fdfd_iga_init -I./$(COVARIANT_DIR) -I./$(CUDA_SOLVER_DIR) \
	  -o $@ $^ $(LDFLAGS_NO_CUDSS)

pml_point_source_demo.o: \
	pml_point_source_demo.cpp \
	fdfd_iga_init/reference_patch_evaluator.hpp \
	covariant_aux_space/covariant_reference_preconditioner.hpp \
	covariant_aux_space/iga_patch_ras_preconditioner.hpp \
	$(CUDA_SOLVER_DIR)/gpu_bicgstab.hpp
	$(CXX) $(CXXFLAGS) -I./fdfd_iga_init -I./$(COVARIANT_DIR) -I./$(CUDA_SOLVER_DIR) \
	  -c pml_point_source_demo.cpp -o $@

# GPU build: adds -solver bicgstab_gpu support (-DHAVE_CUDA, links cuBLAS)
pml_point_source_demo_gpu: \
	pml_point_source_demo.gpu.o \
	$(CUDA_SOLVER_DIR)/gpu_bicgstab.gpu.o \
	$(PML_DEMO_SHARED_OBJS)
	$(CXX) $(CXXFLAGS) -DHAVE_CUDA \
	  -I./fdfd_iga_init -I./$(COVARIANT_DIR) -I./$(CUDA_SOLVER_DIR) $(CUDA_INCLUDE_DIRS) \
	  -o $@ $^ $(GPU_LDFLAGS)

pml_point_source_demo.gpu.o: \
	pml_point_source_demo.cpp \
	fdfd_iga_init/reference_patch_evaluator.hpp \
	covariant_aux_space/covariant_reference_preconditioner.hpp \
	covariant_aux_space/iga_patch_ras_preconditioner.hpp \
	$(CUDA_SOLVER_DIR)/gpu_bicgstab.hpp
	$(CXX) $(CXXFLAGS) -DHAVE_CUDA \
	  -I./fdfd_iga_init -I./$(COVARIANT_DIR) -I./$(CUDA_SOLVER_DIR) $(CUDA_INCLUDE_DIRS) \
	  -c pml_point_source_demo.cpp -o $@

covariant_aux_space/iga_patch_ras_preconditioner.o: \
	covariant_aux_space/iga_patch_ras_preconditioner.cpp \
	covariant_aux_space/iga_patch_ras_preconditioner.hpp
	$(CXX) $(CXXFLAGS) -I./covariant_aux_space -c covariant_aux_space/iga_patch_ras_preconditioner.cpp -o $@

# ── Minimal demo with factory interface ──────────────────────────────
minimal_demo: fdfd_iga_init/minimal_demo.o \
	fdfd_iga_init/reference_fdfd_cpu.o \
	fdfd_iga_init/reference_initial_guess.o \
	covariant_aux_space/yee_transfer.o \
	covariant_aux_space/yee_operator.o \
	covariant_aux_space/covariant_reference_preconditioner.o \
	covariant_aux_space/iga_patch_ras_preconditioner.o
	$(CXX) $(CXXFLAGS) -I./fdfd_iga_init -I./covariant_aux_space -o $@ $^ $(LDFLAGS_NO_CUDSS)

fdfd_iga_init/minimal_demo.o: fdfd_iga_init/minimal_demo.cpp \
	fdfd_iga_init/reference_fdfd_cpu.hpp \
	fdfd_iga_init/reference_patch_evaluator.hpp \
	covariant_aux_space/covariant_preconditioner_factory.hpp \
	covariant_aux_space/iga_patch_ras_preconditioner.hpp
	$(CXX) $(CXXFLAGS) -I./fdfd_iga_init -I./covariant_aux_space -c $< -o $@

# ── PML Split Preconditioner research demo ────────────────────────────────────
SPLIT_DIR = pml_split_preconditioner

pml_split_demo: \
	$(SPLIT_DIR)/pml_split_demo.o \
	$(SPLIT_DIR)/split_pml_prec.o \
	covariant_aux_space/yee_transfer.o \
	covariant_aux_space/yee_operator.o \
	fdfd_iga_init/reference_fdfd_cpu.o
	$(CXX) $(CXXFLAGS) \
	  -I./$(SPLIT_DIR) -I./$(COVARIANT_DIR) -I./fdfd_iga_init \
	  -o $@ $^ $(LDFLAGS_NO_CUDSS)

$(SPLIT_DIR)/pml_split_demo.o: \
	$(SPLIT_DIR)/pml_split_demo.cpp \
	$(SPLIT_DIR)/split_pml_prec.hpp \
	fdfd_iga_init/reference_patch_evaluator.hpp \
	covariant_aux_space/yee_transfer.hpp \
	covariant_aux_space/yee_operator.hpp
	$(CXX) $(CXXFLAGS) \
	  -I./$(SPLIT_DIR) -I./$(COVARIANT_DIR) -I./fdfd_iga_init \
	  -c $< -o $@

$(SPLIT_DIR)/split_pml_prec.o: \
	$(SPLIT_DIR)/split_pml_prec.cpp \
	$(SPLIT_DIR)/split_pml_prec.hpp \
	covariant_aux_space/yee_transfer.hpp \
	covariant_aux_space/yee_operator.hpp \
	fdfd_iga_init/reference_patch_evaluator.hpp
	$(CXX) $(CXXFLAGS) \
	  -I./$(SPLIT_DIR) -I./$(COVARIANT_DIR) -I./fdfd_iga_init \
	  -c $< -o $@

# ── Yee initial guess demo ─────────────────────────────────────────────────
YEE_INIT_DIR = yee_init_guess

yee_init_guess_demo: \
	$(YEE_INIT_DIR)/yee_init_guess_demo.o \
	$(YEE_INIT_DIR)/yee_to_iga_transfer.o \
	covariant_aux_space/yee_operator.o \
	covariant_aux_space/yee_transfer.o \
	fdfd_iga_init/reference_fdfd_cpu.o
	$(CXX) $(CXXFLAGS) -I./$(YEE_INIT_DIR) -I./covariant_aux_space -I./fdfd_iga_init \
	  -o $@ $^ $(LDFLAGS_NO_CUDSS)

$(YEE_INIT_DIR)/yee_init_guess_demo.o: \
	$(YEE_INIT_DIR)/yee_init_guess_demo.cpp \
	$(YEE_INIT_DIR)/yee_to_iga_transfer.hpp \
	covariant_aux_space/yee_operator.hpp \
	fdfd_iga_init/reference_patch_evaluator.hpp
	$(CXX) $(CXXFLAGS) -I./$(YEE_INIT_DIR) -I./covariant_aux_space -I./fdfd_iga_init \
	  -c $< -o $@

$(YEE_INIT_DIR)/yee_to_iga_transfer.o: \
	$(YEE_INIT_DIR)/yee_to_iga_transfer.cpp \
	$(YEE_INIT_DIR)/yee_to_iga_transfer.hpp
	$(CXX) $(CXXFLAGS) -I./$(YEE_INIT_DIR) -I./covariant_aux_space -I./fdfd_iga_init \
	  -c $< -o $@

# ── IGA AMS Preconditioner demo ───────────────────────────────────────────────
IGA_AMS_DIR = iga_ams_preconditioner

iga_ams_demo: \
	$(IGA_AMS_DIR)/iga_ams_demo.o \
	$(IGA_AMS_DIR)/iga_ams_prec.o
	$(CXX) $(CXXFLAGS) -I./$(IGA_AMS_DIR) -I./fdfd_iga_init \
	  -o $@ $^ $(LDFLAGS_NO_CUDSS)

$(IGA_AMS_DIR)/iga_ams_demo.o: \
	$(IGA_AMS_DIR)/iga_ams_demo.cpp \
	$(IGA_AMS_DIR)/iga_ams_prec.hpp \
	fdfd_iga_init/reference_patch_evaluator.hpp
	$(CXX) $(CXXFLAGS) -I./$(IGA_AMS_DIR) -I./fdfd_iga_init \
	  -c $< -o $@

$(IGA_AMS_DIR)/iga_ams_prec.o: \
	$(IGA_AMS_DIR)/iga_ams_prec.cpp \
	$(IGA_AMS_DIR)/iga_ams_prec.hpp
	$(CXX) $(CXXFLAGS) -I./$(IGA_AMS_DIR) \
	  -c $< -o $@

# ── AMS Diagnostics ──────────────────────────────────────────────────────────
ams_diagnostics: \
	$(IGA_AMS_DIR)/ams_diagnostics.o \
	$(IGA_AMS_DIR)/iga_ams_prec.o
	$(CXX) $(CXXFLAGS) -I./$(IGA_AMS_DIR) -I./fdfd_iga_init \
	  -o $@ $^ $(LDFLAGS_NO_CUDSS)

$(IGA_AMS_DIR)/ams_diagnostics.o: \
	$(IGA_AMS_DIR)/ams_diagnostics.cpp \
	fdfd_iga_init/reference_patch_evaluator.hpp
	$(CXX) $(CXXFLAGS) -I./$(IGA_AMS_DIR) -I./fdfd_iga_init \
	  -c $< -o $@

# ── AMS Algebraic Diagnostics ─────────────────────────────────────────────────
ams_algebraic_diag: $(IGA_AMS_DIR)/ams_algebraic_diag.o
	$(CXX) $(CXXFLAGS) -I./$(IGA_AMS_DIR) -I./fdfd_iga_init \
	  -o $@ $^ $(LDFLAGS_NO_CUDSS)

$(IGA_AMS_DIR)/ams_algebraic_diag.o: $(IGA_AMS_DIR)/ams_algebraic_diag.cpp
	$(CXX) $(CXXFLAGS) -I./$(IGA_AMS_DIR) -I./fdfd_iga_init \
	  -c $< -o $@

# ── AMS IGA Coupled Preconditioner ───────────────────────────────────────────
ams_iga_coupled: $(IGA_AMS_DIR)/ams_iga_coupled.o
	$(CXX) $(CXXFLAGS) -I./$(IGA_AMS_DIR) -I./fdfd_iga_init \
	  -o $@ $^ $(LDFLAGS_NO_CUDSS)

$(IGA_AMS_DIR)/ams_iga_coupled.o: $(IGA_AMS_DIR)/ams_iga_coupled.cpp
	$(CXX) $(CXXFLAGS) -I./$(IGA_AMS_DIR) -I./fdfd_iga_init \
	  -c $< -o $@

# ── Q=[G,Pi] Preconditioner Benchmark ─────────────────────────────────────────
ams_q_preconditioner: $(IGA_AMS_DIR)/ams_q_preconditioner.o
	$(CXX) $(CXXFLAGS) -I./$(IGA_AMS_DIR) -I./fdfd_iga_init \
	  -o $@ $^ $(LDFLAGS_NO_CUDSS)

$(IGA_AMS_DIR)/ams_q_preconditioner.o: $(IGA_AMS_DIR)/ams_q_preconditioner.cpp
	$(CXX) $(CXXFLAGS) -I./$(IGA_AMS_DIR) -I./fdfd_iga_init \
	  -c $< -o $@

# ── AMS Parameter Sweep ────────────────────────────────────────────────────────
ams_sweep: $(IGA_AMS_DIR)/ams_sweep.o
	$(CXX) $(CXXFLAGS) -I./$(IGA_AMS_DIR) -I./fdfd_iga_init \
	  -o $@ $^ $(LDFLAGS_NO_CUDSS)

$(IGA_AMS_DIR)/ams_sweep.o: $(IGA_AMS_DIR)/ams_sweep.cpp
	$(CXX) $(CXXFLAGS) -I./$(IGA_AMS_DIR) -I./fdfd_iga_init \
	  -c $< -o $@

# ── AMS Sweep v2 (sparse Pi) ──────────────────────────────────────────────────
ams_sweep_v2: $(IGA_AMS_DIR)/ams_sweep_v2.o
	$(CXX) $(CXXFLAGS) -I./$(IGA_AMS_DIR) -I./fdfd_iga_init \
	  -o $@ $^ $(LDFLAGS_NO_CUDSS)

$(IGA_AMS_DIR)/ams_sweep_v2.o: $(IGA_AMS_DIR)/ams_sweep_v2.cpp
	$(CXX) $(CXXFLAGS) -I./$(IGA_AMS_DIR) -I./fdfd_iga_init \
	  -c $< -o $@

# ── PML Pi_lumped complex preconditioner test ────────────────────────────────
pml_pi_prec_test: pml_pi_prec_test.o
	$(CXX) $(CXXFLAGS) -I./fdfd_iga_init -I./$(COVARIANT_DIR) \
	  -o $@ $^ $(LDFLAGS_NO_CUDSS)

pml_pi_prec_test.o: pml_pi_prec_test.cpp
	$(CXX) $(CXXFLAGS) -I./fdfd_iga_init -I./$(COVARIANT_DIR) \
	  -c $< -o $@

# ── AMS Sweep v3 (fully sparse Pi_lumped) ──────────────────────────────────
ams_sweep_v3: $(IGA_AMS_DIR)/ams_sweep_v3.o
	$(CXX) $(CXXFLAGS) -I./$(IGA_AMS_DIR) -I./fdfd_iga_init \
	  -o $@ $^ $(LDFLAGS_NO_CUDSS)

$(IGA_AMS_DIR)/ams_sweep_v3.o: $(IGA_AMS_DIR)/ams_sweep_v3.cpp
	$(CXX) $(CXXFLAGS) -I./$(IGA_AMS_DIR) -I./fdfd_iga_init \
	  -c $< -o $@

# ── Preconditioner Selector (SPD + PML) ────────────────────────────────────
prec_selector: prec_selector.o
	$(CXX) $(CXXFLAGS) -I./fdfd_iga_init -I./$(COVARIANT_DIR) \
	  -o $@ $^ $(LDFLAGS_NO_CUDSS)

prec_selector.o: prec_selector.cpp
	$(CXX) $(CXXFLAGS) -I./fdfd_iga_init -I./$(COVARIANT_DIR) \
	  -c $< -o $@
