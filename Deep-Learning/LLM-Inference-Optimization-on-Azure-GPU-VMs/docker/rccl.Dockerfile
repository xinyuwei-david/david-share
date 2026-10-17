# Communication benchmark image, separate from the model-serving Dockerfile.
# Source inputs were resolved; this repackaged image has not been built or
# GPU-tested here. The evidence records the earlier direct build on the same
# digest-pinned base, not a deployment receipt for this Dockerfile.
FROM rocm/sgl-dev@sha256:8f8dc55398855764d339ff4e2695b9c3c298704999051acaa1a67993fdff62e5

ARG PIN_RCCL_TESTS_COMMIT=1284e5da11f7904e8a6efd902b9de52253785bd0
RUN test -f /usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h \
 && git init /tmp/rccl-source \
 && git -C /tmp/rccl-source remote add origin https://github.com/ROCm/rocm-systems.git \
 && git -C /tmp/rccl-source sparse-checkout init --cone \
 && git -C /tmp/rccl-source sparse-checkout set projects/rccl-tests \
 && git -C /tmp/rccl-source fetch --depth=1 --filter=blob:none origin "$PIN_RCCL_TESTS_COMMIT" \
 && git -C /tmp/rccl-source checkout --detach FETCH_HEAD \
 && test "$(git -C /tmp/rccl-source rev-parse HEAD)" = "$PIN_RCCL_TESTS_COMMIT" \
 && make -C /tmp/rccl-source/projects/rccl-tests -j8 \
      MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi GPU_TARGETS=gfx942 \
      HIP_HOME=/opt/rocm NCCL_HOME=/opt/rocm \
      GIT_REV="$PIN_RCCL_TESTS_COMMIT" GIT_BRANCH=pinned GIT_DIFF= \
 && install -d /opt/rccl-tests \
 && install -m755 /tmp/rccl-source/projects/rccl-tests/build/all_reduce_perf /opt/rccl-tests/all_reduce_perf \
 && install -d /usr/share/doc/rccl-tests \
 && install -m644 /tmp/rccl-source/projects/rccl-tests/LICENSE.txt /usr/share/doc/rccl-tests/LICENSE.txt \
 && rm -rf /tmp/rccl-source

COPY rccl-entrypoint.sh /usr/local/bin/rccl-entrypoint
RUN chmod 755 /usr/local/bin/rccl-entrypoint
EXPOSE 2222
ENTRYPOINT ["/usr/local/bin/rccl-entrypoint"]
