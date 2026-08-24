# Optimized multi-stage Dockerfile for faster builds
# Stage 1: Dependencies
FROM node:22-slim AS deps
WORKDIR /app

# Install pnpm
RUN npm install -g pnpm@9.15.9 --force --yes

# Copy dependency files and local packages
COPY ./frontend/package.json ./frontend/pnpm-lock.yaml* ./
COPY ./frontend/vendor/ ./vendor/

# Install dependencies with cache mount for faster rebuilds
RUN --mount=type=cache,target=/root/.local/share/pnpm/store \
    pnpm config set auto-install-peers true && \
    if [ -f pnpm-lock.yaml ]; then \
        pnpm install --no-optional --frozen-lockfile || pnpm install --no-optional --no-frozen-lockfile; \
    else \
        pnpm install --no-optional --no-frozen-lockfile; \
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

# Build with cache mount for Next.js cache
RUN --mount=type=cache,target=/app/.next/cache \
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
