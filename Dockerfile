FROM rust:alpine
WORKDIR /app 
COPY . .
RUN apk add --no-cache musl-dev gcc g++ make
RUN cargo build --release
RUN rm -rf /usr/local/cargo/registry && \
    rm -rf /usr/local/cargo/git
CMD ["cargo","test","--release"]
