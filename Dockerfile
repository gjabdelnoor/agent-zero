# syntax=docker/dockerfile:1

# Build the runtime image for Agent Zero using the published base image.
FROM agent0ai/agent-zero-base:latest

# Default to the "local" installation mode which leverages the checked out
# repository content that GitHub Actions provides via actions/checkout.
ARG BRANCH=local
ENV BRANCH=${BRANCH}

# Copy the runtime filesystem overlays used by the installer scripts.
COPY ./docker/run/fs/ /

# Copy the current repository so the installation scripts can reference it.
COPY ./ /git/agent-zero

# Execute the standard installation pipeline that is used across the project.
RUN bash /ins/pre_install.sh ${BRANCH} \
    && bash /ins/install_A0.sh ${BRANCH} \
    && bash /ins/install_additional.sh ${BRANCH}

# Cache buster support for GitHub Actions builds.
ARG CACHE_DATE=none
RUN echo "cache buster ${CACHE_DATE}" \
    && bash /ins/install_A02.sh ${BRANCH}

# Finalise the container image.
RUN bash /ins/post_install.sh ${BRANCH}

# Expose the primary ports that the runtime needs.
EXPOSE 22 80 9000-9009

# Ensure entrypoint scripts are executable and define default command.
RUN chmod +x /exe/initialize.sh /exe/run_A0.sh /exe/run_searxng.sh /exe/run_tunnel_api.sh
CMD ["/bin/sh", "-c", "/exe/initialize.sh \"$BRANCH\""]
