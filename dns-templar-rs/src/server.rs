use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::UdpSocket;
use tokio::time::timeout;
use hickory_proto::op::{Message};
use hickory_proto::serialize::binary::{BinDecodable};

use crate::classifier::DnsTemplar;

pub async fn serve(
    listen_addr: &str,
    upstream_addr: &str,
    templar: Arc<DnsTemplar>,
    threshold: Option<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let socket = Arc::new(UdpSocket::bind(listen_addr).await?);
    let upstream: SocketAddr = upstream_addr.parse()?;

    tracing::info!("dns-templar listening on {listen_addr}, forwarding clean queries to {upstream_addr}");

    loop {
        let mut buf = [0u8; 4096];
        let (len, src) = socket.recv_from(&mut buf).await?;

        let socket = Arc::clone(&socket);
        let templar = Arc::clone(&templar);

        tokio::spawn(async move {
            if let Err(e) = handle_query(&socket, &buf[..len], src, &templar, upstream, threshold).await {
                tracing::warn!("query error from {src}: {e}");
            }
        });
    }
}

async fn handle_query(
    socket: &UdpSocket,
    raw: &[u8],
    src: SocketAddr,
    templar: &Arc<DnsTemplar>,
    upstream: SocketAddr,
    threshold: Option<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let message = Message::from_bytes(raw)?;

    let domain = match message.queries.first() {
        Some(q) => q.name().to_string(),
        None => return Ok(()),
    };

    let templar = Arc::clone(templar);
    let verdict = tokio::task::spawn_blocking(move || {
        templar.classify(&domain, threshold)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    tracing::info!(
        domain = %verdict.domain,
        probability = verdict.probability,
        is_dga = verdict.is_dga,
        whitelisted = verdict.whitelisted,
        tier = ?verdict.tier,
        "classified"
    );

    if verdict.is_dga && !verdict.whitelisted {
        // tracing::info!("blocking DGA, building NXDOMAIN");
        let response = build_nxdomain(raw);
        // tracing::info!("NXDOMAIN built ({} bytes), sending to client", response.len());
        socket.send_to(&response, src).await?;
        // tracing::info!("NXDOMAIN sent");

    } else {
        // tracing::info!("forwarding to upstream");
        let upstream_sock = UdpSocket::bind("0.0.0.0:0").await?;
        upstream_sock.send_to(raw, upstream).await?;
        let mut resp_buf = [0u8; 4096];
        let (resp_len, _) = timeout(Duration::from_secs(2), upstream_sock.recv_from(&mut resp_buf))
            .await
            .map_err(|_| format!("upstream {upstream} timed out"))??;
        socket.send_to(&resp_buf[..resp_len], src).await?;
    }

    Ok(())
}

fn build_nxdomain(raw_query: &[u8]) -> Vec<u8> {
    let mut r = raw_query.to_vec();
    if r.len() >= 4 {
        r[2] |= 0x80;                   // QR = 1 (response)
        r[3] = (r[3] & 0xF0) | 0x03;   // RCODE = 3 (NXDOMAIN)
    }
    r
}