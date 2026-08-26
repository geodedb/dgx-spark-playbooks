# Optimized multi-stage Dockerfile for faster builds
# Stage 1: Dependencies
FROM node:22-slim AS deps
WORKDIR /app

# Install pnpm
RUN npm install -g pnpm@9.15.9 --force --yes

# Copy dependency files and local packages
COPY ./frontend/package.json ./frontend/pnpm-lock.yaml* ./
COPY ./frontend/vendor/ ./vendor/

# Install dependencies with cache mount for faster rebuilds.
# NOTE: optionalDependencies MUST be installed (no --no-optional): the build needs
# platform-specific native prebuilds such as lightningcss-linux-x64-gnu (Tailwind 4 /
# Turbopack) and @matrixai/quic-linux-x64 (QUIC transport for @geodedb/client).
# Cache mount id is versioned (v2) to bust stores poisoned by earlier --no-optional installs.
RUN --mount=type=cache,id=pnpm-store-v2,target=/root/.local/share/pnpm/store \
    pnpm config set auto-install-peers true && \
    if [ -f pnpm-lock.yaml ]; then \
        pnpm install --frozen-lockfile || pnpm install --no-frozen-lockfile; \
    else \
        pnpm install --no-frozen-lockfile; \
    fi

# Stage 2: Builder
FROM node:22-slim AS builder
WORKDIR /app

# Install pnpm
RUN npm install -g pnpm@9.15.9 --force --yes

# Copy node_modules from deps stage
COPY --from=deps /app/node_modules ./node_modules
COPY --from=deps /app/package.json ./package.json

# Copy source code
COPY ./frontend/ ./

# Set build environment variables
ENV NEXT_TELEMETRY_DISABLED 1
ENV NODE_ENV production

# Build with cache mount for Next.js cache.
# Cache mount id is versioned (v2) to bust Turbopack caches poisoned by builds
# that ran before optionalDependencies were installed (missing lightningcss binary).
RUN --mount=type=cache,id=next-cache-v2,target=/app/.next/cache \
    pnpm build

# Stage 3: Production runtime
FROM node:22-slim AS runner
WORKDIR /app

ENV NODE_ENV production
ENV NEXT_TELEMETRY_DISABLED 1

# Create non-root user for security
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Copy necessary files from builder
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Fix: Copy proto file for @geodedb/client (Next.js standalone doesn't trace dynamic file loads)
COPY --chown=nextjs:nodejs ./frontend/vendor/geode-proto/geode.proto /tmp/geode.proto
RUN find /app/node_modules -path "*/@geodedb/client" -type d -exec mkdir -p {}/proto \; -exec cp /tmp/geode.proto {}/proto/ \; && rm /tmp/geode.proto

# Fix: Next.js standalone output tracing doesn't follow @matrixai/quic's dynamic
# native-binding require, so the linux-x64 prebuild is missing at runtime and every
# QUIC connection fails ("Failed requiring possible native bindings"). Copy the
# prebuild package (version pinned by pnpm-lock.yaml) into the runner's node_modules.
COPY --from=deps --chown=nextjs:nodejs \
    /app/node_modules/.pnpm/@matrixai+quic-linux-x64@2.0.9/node_modules/@matrixai/quic-linux-x64 \
    ./node_modules/@matrixai/quic-linux-x64

# Create data directory for query logs with proper permissions
RUN mkdir -p /app/data && chown -R nextjs:nodejs /app/data

USER nextjs

EXPOSE 3000

ENV PORT 3000
ENV HOSTNAME "0.0.0.0"

# Matches the docker-compose healthcheck already defined for this service.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD node -e "require('http').get('http://localhost:3000/', r => process.exit(r.statusCode < 500 ? 0 : 1)).on('error', () => process.exit(1))"

CMD ["node", "server.js"]
