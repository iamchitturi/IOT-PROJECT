#include <WiFi.h>
#include <HTTPClient.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include <time.h>

const char* WIFI_SSID     = "Chaitanya";
const char* WIFI_PASSWORD = "chchaitu";

const char* THINGSPEAK_SERVER = "http://api.thingspeak.com";
const char* WRITE_API_KEY     = "PICVJCVLA31J1KJK";
const long  CHANNEL_ID        = 3286342;

#define TDS_PIN        32
#define LED_PIN        2
#define VREF           3.3
#define ADC_RESOLUTION 4096.0
#define NUM_SAMPLES    20
#define UPLOAD_INTERVAL 20000

unsigned long lastUpload = 0;
int uploadCount = 0;

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("================================");
  Serial.println("  TDS & Temp Sensor Monitor");
  Serial.println("================================");
  Serial.println();

  pinMode(TDS_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    attempts++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("WiFi Connected! IP: ");
    Serial.println(WiFi.localIP());
    digitalWrite(LED_PIN, HIGH);
    
    // Sync time for temperature diurnal simulation (IST: +5:30 = 19800 sec)
    configTime(19800, 0, "pool.ntp.org");
  } else {
    Serial.println("WiFi FAILED! Check credentials.");
    digitalWrite(LED_PIN, LOW);
  }

  Serial.println();
  Serial.println("Starting readings...");
  Serial.println();
}

void loop() {
  if (millis() - lastUpload >= UPLOAD_INTERVAL || lastUpload == 0) {
    lastUpload = millis();

    // ── TDS Reading ──
    int readings[NUM_SAMPLES];
    for (int i = 0; i < NUM_SAMPLES; i++) {
      readings[i] = analogRead(TDS_PIN);
      delay(20);
    }

    for (int i = 0; i < NUM_SAMPLES - 1; i++) {
      for (int j = 0; j < NUM_SAMPLES - i - 1; j++) {
        if (readings[j] > readings[j + 1]) {
          int t = readings[j];
          readings[j] = readings[j + 1];
          readings[j + 1] = t;
        }
      }
    }

    float avg = 0;
    int s = NUM_SAMPLES * 0.2;
    int e = NUM_SAMPLES * 0.8;
    for (int i = s; i < e; i++) avg += readings[i];
    avg /= (e - s);

    float voltage = avg * VREF / ADC_RESOLUTION;
    float tds = (133.42 * voltage * voltage * voltage
               - 255.86 * voltage * voltage
               + 857.39 * voltage) * 0.5;
    if (tds < 0) tds = 0;

    int qualityIndex = 0;
    String qualityLabel = "";

    if ((int)avg < 10) {
      qualityLabel = "SENSOR NOT IN WATER";
      qualityIndex = 0;
    } else if (tds < 50) {
      qualityLabel = "SAFE - Pure/Very Clean";
      qualityIndex = 5;
    } else if (tds < 200) {
      qualityLabel = "SAFE - Good Drinking Water (WHO OK)";
      qualityIndex = 4;
    } else if (tds < 400) {
      qualityLabel = "MODERATE - Exceeds WHO Limit";
      qualityIndex = 3;
    } else if (tds < 600) {
      qualityLabel = "POOR - Not Safe for Drinking";
      qualityIndex = 2;
    } else {
      qualityLabel = "UNSAFE - Harmful! Do Not Drink";
      qualityIndex = 1;
    }

    // ── Simulated Temperature Reading (26–33 °C) ──
    // Varies realistically and continuously so the user can see immediate changes on ThingSpeak
    float temperature = 29.5; 
    
    // Cycle every ~20 minutes for testing visibility (instead of 24h diurnal)
    float time_val = (millis() % 1200000) / 1200000.0; // 0.0 to 1.0
    float diurnal = sin(time_val * 2.0 * PI);
    temperature = 29.5 + diurnal * 3.5;
    
    // Add tiny atmospheric random noise ±0.2°C
    temperature += random(-4, 5) / 20.0;
    
    // Clamp exactly to 26.0 - 33.0 limits 
    if (temperature < 26.0) temperature = 26.0 + random(0, 3) / 10.0;
    if (temperature > 33.0) temperature = 33.0 - random(0, 3) / 10.0;

    Serial.println("----------------------------------------");
    Serial.print("ADC: ");
    Serial.print((int)avg);
    Serial.print(" | Voltage: ");
    Serial.print(voltage, 3);
    Serial.print("V | TDS: ");
    Serial.print(tds, 1);
    Serial.print(" ppm | Temp: ");
    Serial.print(temperature, 2);
    Serial.println(" °C");
    Serial.print("Quality: [");
    Serial.print(qualityIndex);
    Serial.print("] ");
    Serial.println(qualityLabel);
    Serial.print("WiFi: ");
    if (WiFi.status() == WL_CONNECTED) {
      Serial.print("Connected (");
      Serial.print(WiFi.RSSI());
      Serial.println(" dBm)");
    } else {
      Serial.println("Disconnected");
    }
    Serial.print("Uploads: ");
    Serial.println(uploadCount);
    Serial.println("----------------------------------------");

    if (WiFi.status() == WL_CONNECTED) {
      WiFiClient client;
      HTTPClient http;

      // Upload TDS to field1, Voltage to field2, Quality to field3, Temp to field4
      String url = String(THINGSPEAK_SERVER) + "/update?api_key=" + WRITE_API_KEY;
      url += "&field1=" + String(tds, 2);
      url += "&field2=" + String(voltage, 3);
      url += "&field3=" + String(qualityIndex);
      url += "&field4=" + String(temperature, 2);

      Serial.print("Uploading... ");

      http.begin(client, url);
      http.setTimeout(10000);
      int httpCode = http.GET();

      if (httpCode > 0) {
        String response = http.getString();
        int entryId = response.toInt();
        if (entryId > 0) {
          uploadCount++;
          Serial.print("OK! Entry #");
          Serial.println(entryId);
          for (int i = 0; i < 2; i++) {
            digitalWrite(LED_PIN, LOW);  delay(150);
            digitalWrite(LED_PIN, HIGH); delay(150);
          }
        } else {
          Serial.println("ThingSpeak returned 0 (rate limit?)");
        }
      } else {
        Serial.print("HTTP Error: ");
        Serial.println(httpCode);
      }
      http.end();
    } else {
      Serial.println("WiFi disconnected, trying reconnect...");
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    }

    Serial.println();
  }

  delay(100);
}
