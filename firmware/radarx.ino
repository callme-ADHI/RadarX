/**
 * RadarX v2.0 — SoftAP + Promiscuous CSI Radar
 * ===============================================
 * Hardware : ESP32 DevKit V1, CP2102 USB bridge
 * Core     : esp32 by Espressif Systems 3.3.10
 * Baud     : 921600
 *
 * Architecture (no router needed):
 *   1. Start SoftAP "RadarX-Net" on channel 6.
 *   2. Enable promiscuous mode — capture all 802.11 frames.
 *   3. Enable CSI collection — extract HT-LTF per captured frame.
 *   4. Every 100 ms send a synthetic beacon via esp_wifi_80211_tx()
 *      to maintain a steady stream of frames even with no clients.
 *   5. CSI callback prints CSV line to Serial @ 921600.
 *   6. Listen for "LED:ON\n" / "LED:OFF\n" commands from PC.
 *
 * Serial output:
 *   CSI_DATA,<ms>,<rssi>,<noise>,<ch>,<n_sub>,[I0,Q0,I1,Q1,...]
 *   HEARTBEAT,<ms>,<free_heap>
 */

#include <Arduino.h>
#include <WiFi.h>
#include "esp_wifi.h"
#include "esp_wifi_types.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// ─── Configuration ───────────────────────────────────────────────────────────
static const char*    AP_SSID      = "RadarX-Net";
static const char*    AP_PASSWORD  = "radarx123";
static const uint8_t  WIFI_CHANNEL = 6;
static const uint32_t SERIAL_BAUD  = 921600;
static const uint32_t BEACON_INTERVAL_MS = 100;
static const int      LED_PIN      = 2;    // Built-in LED on most DevKits
static const int      MAX_CSI_BYTES = 128; // 64 I/Q pairs → 52 data subcarriers

// ─── Globals ─────────────────────────────────────────────────────────────────
static volatile bool ledHuman = false;
static uint32_t      lastBeaconMs = 0;
static uint32_t      lastHeartbeatMs = 0;
static uint32_t      lastLedToggleMs = 0;
static bool          ledState = false;

// ─── Synthetic beacon frame (minimal 802.11 beacon) ──────────────────────────
// This keeps CSI flowing even when no real client is associated.
// Destination: broadcast, Source & BSSID: RadarX MAC, fixed values.
static const uint8_t BEACON_FRAME[] = {
    // Frame Control: beacon (type=0, subtype=8)
    0x80, 0x00,
    // Duration
    0x00, 0x00,
    // Destination: broadcast
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    // Source MAC (placeholder — overwritten at runtime)
    0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
    // BSSID (same as source)
    0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
    // Sequence number (will increment)
    0x00, 0x00,
    // Timestamp (8 bytes, zeroed — AP fills real value)
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    // Beacon interval: 100 TU
    0x64, 0x00,
    // Capability info: ESS + short preamble
    0x31, 0x04,
    // SSID IE
    0x00, 0x0A, 'R','a','d','a','r','X','-','N','e','t',
    // Supported rates IE: 1, 2, 5.5, 11 Mbps
    0x01, 0x04, 0x82, 0x84, 0x8B, 0x96,
    // DS Parameter Set (channel 6)
    0x03, 0x01, 0x06,
};

// ─── CSI callback ─────────────────────────────────────────────────────────────
/**
 * Called by WiFi driver for every received 802.11 frame when CSI is enabled.
 * Formats raw CSI bytes as CSV and prints to Serial.
 *
 * Format:
 *   CSI_DATA,<ms>,<rssi>,<noise>,<channel>,<n_bytes>,[b0,b1,...]
 */
static void IRAM_ATTR csi_rx_callback(void* ctx, wifi_csi_info_t* info)
{
    if (!info || !info->buf || info->len == 0) return;

    uint32_t ts     = (uint32_t)millis();
    int8_t   rssi   = info->rx_ctrl.rssi;
    int8_t   noise  = info->rx_ctrl.noise_floor;
    uint8_t  ch     = info->rx_ctrl.channel;
    int      len    = (info->len > MAX_CSI_BYTES) ? MAX_CSI_BYTES : info->len;

    // Build line into stack buffer — no heap allocation in ISR context
    // Worst case: "CSI_DATA,4294967295,-128,-128,13,128," + 128*(-128+",") = ~700 chars
    char line[800];
    int  pos = 0;

    pos += snprintf(line + pos, sizeof(line) - pos,
                    "CSI_DATA,%lu,%d,%d,%u,%d,[",
                    (unsigned long)ts, (int)rssi, (int)noise, (unsigned)ch, len);

    for (int i = 0; i < len && pos < (int)sizeof(line) - 6; i++) {
        if (i > 0) line[pos++] = ',';
        pos += snprintf(line + pos, sizeof(line) - pos, "%d", (int)info->buf[i]);
    }

    if (pos < (int)sizeof(line) - 3) {
        line[pos++] = ']';
        line[pos++] = '\n';
        line[pos]   = '\0';
    }

    Serial.write((uint8_t*)line, pos);
}

// ─── Promiscuous callback (required to keep promisc mode active) ──────────────
static void IRAM_ATTR promisc_callback(void* buf, wifi_promiscuous_pkt_type_t type)
{
    // We use the CSI callback for data — this is a no-op placeholder
    (void)buf;
    (void)type;
}

// ─── Enable CSI collection ────────────────────────────────────────────────────
static void enable_csi()
{
    wifi_csi_config_t csi_cfg;
    memset(&csi_cfg, 0, sizeof(csi_cfg));
    csi_cfg.lltf_en           = true;
    csi_cfg.htltf_en          = true;
    csi_cfg.stbc_htltf2_en    = true;
    csi_cfg.ltf_merge_en      = true;
    csi_cfg.channel_filter_en = false;  // raw, unfiltered
    csi_cfg.manu_scale        = false;

    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_cfg));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(csi_rx_callback, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
    Serial.println("CSI_ENABLED");
}

// ─── Send synthetic beacon ───────────────────────────────────────────────────
static void send_beacon()
{
    static uint16_t seq = 0;

    // Copy frame and patch source/BSSID with real MAC
    uint8_t frame[sizeof(BEACON_FRAME)];
    memcpy(frame, BEACON_FRAME, sizeof(BEACON_FRAME));

    uint8_t mac[6];
    esp_wifi_get_mac(WIFI_IF_AP, mac);
    memcpy(frame + 10, mac, 6);  // Source
    memcpy(frame + 16, mac, 6);  // BSSID

    // Patch sequence number
    frame[22] = (seq << 4) & 0xF0;
    frame[23] = (seq >> 4) & 0xFF;
    seq++;

    // Transmit on AP interface, no ACK needed
    esp_wifi_80211_tx(WIFI_IF_AP, frame, sizeof(frame), false);
}

// ─── setup() ─────────────────────────────────────────────────────────────────
void setup()
{
    Serial.begin(SERIAL_BAUD);
    delay(200);

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    Serial.println("RADARX_BOOT_V2");

    // ── WiFi: SoftAP mode ────────────────────────────────────────────────────
    WiFi.mode(WIFI_AP);

    // Force 802.11n HT20 on AP interface for HT-LTF CSI
    esp_wifi_set_protocol(WIFI_IF_AP,
        WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);

    WiFi.softAP(AP_SSID, AP_PASSWORD, WIFI_CHANNEL, 0 /*hidden=false*/, 4 /*max_conn*/);

    Serial.printf("SOFTAP_STARTED,SSID:%s,CH:%d,IP:%s\n",
                  AP_SSID, WIFI_CHANNEL,
                  WiFi.softAPIP().toString().c_str());

    // ── Promiscuous mode (required for CSI on AP interface) ──────────────────
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_promiscuous_rx_cb(promisc_callback);

    // Filter: capture management + data frames (not control)
    wifi_promiscuous_filter_t filter;
    filter.filter_mask = WIFI_PROMIS_FILTER_MASK_MGMT | WIFI_PROMIS_FILTER_MASK_DATA;
    esp_wifi_set_promiscuous_filter(&filter);

    // ── CSI ──────────────────────────────────────────────────────────────────
    enable_csi();

    Serial.println("RADARX_READY");
}

// ─── loop() ──────────────────────────────────────────────────────────────────
void loop()
{
    uint32_t now = millis();

    // ── Send synthetic beacon every 100 ms ───────────────────────────────────
    if (now - lastBeaconMs >= BEACON_INTERVAL_MS) {
        lastBeaconMs = now;
        send_beacon();
    }

    // ── Heartbeat every 1000 ms ───────────────────────────────────────────────
    if (now - lastHeartbeatMs >= 1000) {
        lastHeartbeatMs = now;
        Serial.printf("HEARTBEAT,%lu,%u\n",
                      (unsigned long)now,
                      (unsigned)esp_get_free_heap_size());
    }

    // ── LED control ───────────────────────────────────────────────────────────
    if (ledHuman) {
        // Solid ON when human detected
        digitalWrite(LED_PIN, HIGH);
    } else {
        // Blink at 1 Hz (normal operation)
        if (now - lastLedToggleMs >= 500) {
            lastLedToggleMs = now;
            ledState = !ledState;
            digitalWrite(LED_PIN, ledState ? HIGH : LOW);
        }
    }

    // ── Serial command handling (from PC) ─────────────────────────────────────
    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        if (cmd == "LED:ON") {
            ledHuman = true;
        } else if (cmd == "LED:OFF") {
            ledHuman = false;
        } else if (cmd == "REBOOT") {
            ESP.restart();
        } else if (cmd == "STATUS") {
            Serial.printf("STATUS,HEAP:%u,UPTIME:%lu,CH:%d\n",
                          (unsigned)esp_get_free_heap_size(),
                          (unsigned long)now, WIFI_CHANNEL);
        }
    }

    delay(5);  // yield to WiFi task, prevent WDT
}
