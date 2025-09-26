# Computer-use focused runtime image built directly from the repository
FROM agent0ai/agent-zero-base:latest

# Use local sources by default; override with BRANCH to pull from Git
ARG BRANCH=local
ENV BRANCH=$BRANCH

# Set default subordinate profile for interactive sessions (optional override at runtime)
ENV AGENT_ZERO_DEFAULT_PROFILE=computer_use

# Copy runtime filesystem overlay
COPY ./docker/run/fs/ /

# Ship the current workspace for local builds
COPY . /git/agent-zero

# Prepare and install Agent Zero inside the container
RUN bash /ins/pre_install.sh ${BRANCH} && \
    bash /ins/install_A0.sh ${BRANCH} && \
    bash /ins/install_additional.sh ${BRANCH}

# Cache buster for rebuilds without stale artifacts
ARG CACHE_DATE=none
RUN echo "cache buster ${CACHE_DATE}" && bash /ins/install_A02.sh ${BRANCH}

# Finalize installation and clean up
RUN bash /ins/post_install.sh ${BRANCH}

# Expose ssh, web ui, tunnel, and desktop bridges
EXPOSE 22 80 5901 6080 9000-9009

# Ensure runtime entrypoints are executable
RUN chmod +x /exe/initialize.sh /exe/run_A0.sh /exe/run_searxng.sh /exe/run_tunnel_api.sh \
    /exe/run_desktop_xvfb.sh /exe/run_desktop_session.sh /exe/run_desktop_vnc.sh /exe/run_desktop_novnc.sh

# Launch supervisor-managed stack
CMD ["/exe/initialize.sh", "$BRANCH"]
