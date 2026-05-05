FROM node:20-bookworm-slim

ENV NEXT_TELEMETRY_DISABLED=1

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 3000

# Build on container start so NEXT_PUBLIC_* values from env_file are available
# without maintaining a second Compose-only env source.
CMD ["sh", "-c", "npm run build && ./node_modules/.bin/next start --hostname 0.0.0.0 --port 3000"]
