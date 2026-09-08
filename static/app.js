// ==========================================
// NER LANDSLIDE COMMAND CENTER
// Frontend for converted_app.py
// ==========================================

const API = "";

let appConfig = null;
let gisMap = null;
let gisLayer = null;
let lastPrediction = null;
let chatHistory = [];

const $ = (id) => document.getElementById(id);


// ==========================================
// GENERIC HELPERS
// ==========================================

function setText(id, value) {
    const element = $(id);

    if (element) {
        element.textContent =
            value === null ||
            value === undefined ||
            value === ""
                ? "--"
                : value;
    }
}


function setMessage(
    id,
    message = "",
    type = ""
) {
    const element = $(id);

    if (!element) {
        return;
    }

    element.textContent = message;
    element.className =
        `message ${type}`.trim();
}


function show(id) {
    const element = $(id);

    if (element) {
        element.classList.remove("hidden");
    }
}


function hide(id) {
    const element = $(id);

    if (element) {
        element.classList.add("hidden");
    }
}


function numberValue(id) {
    return Number(
        $(id)?.value || 0
    );
}


function formatNumber(
    value,
    digits = 1
) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "--";
    }

    return number.toFixed(digits);
}


function escapeHTML(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


async function readResponse(response) {
    let data;

    try {
        data = await response.json();
    }
    catch {
        throw new Error(
            `Server returned invalid response (${response.status}).`
        );
    }

    if (!response.ok) {
        throw new Error(
            data.detail ||
            data.message ||
            `Request failed (${response.status}).`
        );
    }

    return data;
}


async function apiFetch(
    url,
    options = {}
) {
    const response = await fetch(
        `${API}${url}`,
        options
    );

    return readResponse(
        response
    );
}


function setButtonLoading(
    button,
    loading,
    normalText,
    loadingText
) {
    if (!button) {
        return;
    }

    button.disabled = loading;
    button.textContent =
        loading
            ? loadingText
            : normalText;
}


// ==========================================
// CLOCK
// ==========================================

function updateClock() {
    setText(
        "currentTime",
        new Date().toLocaleString(
            "en-IN",
            {
                dateStyle: "medium",
                timeStyle: "medium"
            }
        )
    );
}


// ==========================================
// NAVIGATION
// ==========================================

function openPage(pageName) {
    document
        .querySelectorAll(".page")
        .forEach(
            (page) => {
                page.classList.remove("active");
            }
        );

    document
        .querySelectorAll(".nav-item")
        .forEach(
            (item) => {
                item.classList.remove("active");
            }
        );

    const page =
        $(`page-${pageName}`);

    if (page) {
        page.classList.add("active");
    }

    const nav =
        document.querySelector(
            `[data-page="${pageName}"]`
        );

    if (nav) {
        nav.classList.add("active");
    }

    $("sidebar")?.classList.remove("open");

    if (
        pageName === "gis" &&
        gisMap
    ) {
        setTimeout(
            () => gisMap.invalidateSize(),
            150
        );
    }

    if (pageName === "citizen") {
        loadReports();
    }

    if (pageName === "analytics") {
        loadAnalytics();
    }
}


function initializeNavigation() {
    document
        .querySelectorAll(".nav-item")
        .forEach(
            (item) => {
                item.addEventListener(
                    "click",
                    () => {
                        openPage(
                            item.dataset.page
                        );
                    }
                );
            }
        );

    $("menuButton")?.addEventListener(
        "click",
        () => {
            $("sidebar")?.classList.toggle(
                "open"
            );
        }
    );
}


// ==========================================
// CONFIG / STATES
// ==========================================

function fillStateSelect(
    selectId,
    includeAll = false
) {
    const select = $(selectId);

    if (
        !select ||
        !appConfig
    ) {
        return;
    }

    select.innerHTML = "";

    if (includeAll) {
        const option =
            document.createElement("option");

        option.value =
            "All NER States";

        option.textContent =
            "All NER States";

        select.appendChild(
            option
        );
    }

    appConfig.states.forEach(
        (state) => {
            const option =
                document.createElement("option");

            option.value = state;
            option.textContent = state;

            select.appendChild(
                option
            );
        }
    );
}


async function loadConfig() {
    appConfig = await apiFetch(
        "/api/config"
    );

    fillStateSelect(
        "predictionState"
    );

    fillStateSelect(
        "gisState",
        true
    );

    fillStateSelect(
        "reportState"
    );

    fillStateSelect(
        "weatherState"
    );

    fillStateSelect(
        "analyticsState",
        true
    );

    updatePredictionDefaultCity();
    updateReportCoordinates();
    updateWeatherCities();
}


function updatePredictionDefaultCity() {
    if (!appConfig) {
        return;
    }

    const state =
        $("predictionState")?.value;

    if (!state) {
        return;
    }

    const city =
        appConfig.default_cities[
            state
        ];

    if ($("predictionCity")) {
        $("predictionCity").value =
            city || "";
    }
}


function updateReportCoordinates() {
    if (!appConfig) {
        return;
    }

    const state =
        $("reportState")?.value;

    const coordinates =
        appConfig.default_coordinates[
            state
        ];

    if (!coordinates) {
        return;
    }

    $("reportLatitude").value =
        coordinates.latitude;

    $("reportLongitude").value =
        coordinates.longitude;
}



// ==========================================
// LIVE WEATHER STATE / CITY OPTIONS
// ==========================================

const WEATHER_CITIES = {
    "Assam": [
        "Haflong",
        "Guwahati",
        "Silchar",
        "Dibrugarh",
        "Jorhat",
        "Tezpur"
    ],
    "Arunachal Pradesh": [
        "Itanagar",
        "Tawang",
        "Bomdila",
        "Pasighat",
        "Ziro",
        "Naharlagun"
    ],
    "Manipur": [
        "Imphal",
        "Ukhrul",
        "Churachandpur",
        "Senapati",
        "Tamenglong"
    ],
    "Meghalaya": [
        "Shillong",
        "Cherrapunji",
        "Mawsynram",
        "Tura",
        "Jowai",
        "Nongpoh"
    ],
    "Mizoram": [
        "Aizawl",
        "Lunglei",
        "Champhai",
        "Kolasib",
        "Serchhip"
    ],
    "Nagaland": [
        "Kohima",
        "Dimapur",
        "Mokokchung",
        "Wokha",
        "Tuensang"
    ],
    "Sikkim": [
        "Gangtok",
        "Namchi",
        "Gyalshing",
        "Mangan",
        "Rangpo"
    ],
    "Tripura": [
        "Agartala",
        "Dharmanagar",
        "Udaipur",
        "Kailashahar",
        "Belonia"
    ]
};


function updateWeatherCities() {

    const stateSelect = $("weatherState");
    const citySelect = $("weatherCity");

    if (!stateSelect || !citySelect) {
        return;
    }

    const state = stateSelect.value;
    const cities = WEATHER_CITIES[state] || [];

    citySelect.innerHTML = "";

    cities.forEach((city) => {

        const option =
            document.createElement("option");

        option.value = city;
        option.textContent = city;

        citySelect.appendChild(
            option
        );
    });

    // Keep the backend's configured default city selected first.
    const defaultCity =
        appConfig?.default_cities?.[state];

    if (
        defaultCity &&
        cities.includes(defaultCity)
    ) {
        citySelect.value = defaultCity;
    }
}


// ==========================================
// HEALTH / OVERVIEW
// ==========================================

function healthText(
    ready,
    readyText = "Ready",
    badText = "Check configuration"
) {
    return ready
        ? `🟢 ${readyText}`
        : `🟡 ${badText}`;
}


async function loadHealth() {
    try {
        const data =
            await apiFetch(
                "/api/health"
            );

        const services =
            data.services || {};

        setText(
            "healthML",
            healthText(
                services.ml_engine?.ready
            )
        );

        setText(
            "healthGIS",
            healthText(
                services.gis?.ready
            )
        );

        setText(
            "healthWeather",
            healthText(
                services.weather?.ready,
                "Configured",
                "API key missing"
            )
        );

        setText(
            "healthEmail",
            healthText(
                services.email?.ready,
                "Configured",
                "Credentials missing"
            )
        );

        setText(
            "healthGenAI",
            healthText(
                services.genai?.ready,
                "Configured",
                "API key missing"
            )
        );

        const network =
            data.network || {};

        setText(
            "sidebarNetwork",
            network.online
                ? "Online"
                : "Offline / Low Network"
        );

        setText(
            "pendingReports",
            network.pending_reports ?? 0
        );

        setText(
            "pendingAlerts",
            network.pending_alerts ?? 0
        );
    }
    catch (error) {
        setText(
            "sidebarNetwork",
            "Backend unavailable"
        );
    }
}


async function loadOverview() {
    try {
        const data =
            await apiFetch(
                "/api/overview"
            );

        setText(
            "overviewStates",
            data.states
        );

        setText(
            "overviewEvents",
            data.historical_events
        );

        setText(
            "overviewSamples",
            data.training_samples
        );

        await loadHealth();
    }
    catch (error) {
        console.error(
            "Overview error:",
            error
        );
    }
}


// ==========================================
// RISK PREDICTION
// ==========================================

function updateRangeLabels() {
    setText(
        "soil1Value",
        Number(
            $("soil1").value
        ).toFixed(2)
    );

    setText(
        "soil2Value",
        Number(
            $("soil2").value
        ).toFixed(2)
    );
}


function setRiskBadge(
    level
) {
    const badge =
        $("predictionRiskBadge");

    if (!badge) {
        return;
    }

    const risk =
        String(
            level || "MODERATE"
        ).toUpperCase();

    badge.textContent =
        `${risk} RISK`;

    badge.className =
        `risk-badge ${risk.toLowerCase()}`;
}


function renderRiskDrivers(
    drivers
) {
    const container =
        $("riskDrivers");

    if (!container) {
        return;
    }

    if (
        !Array.isArray(drivers) ||
        drivers.length === 0
    ) {
        container.innerHTML =
            `<div class="driver-card">
                Combined environmental conditions.
             </div>`;

        return;
    }

    container.innerHTML =
        drivers.map(
            (driver) => `
                <div class="driver-card">
                    <strong>
                        ${escapeHTML(driver.title)}
                    </strong>
                    <br>
                    ${escapeHTML(driver.detail)}
                </div>
            `
        ).join("");
}


function objectSummary(
    object,
    preferredKeys = []
) {
    if (
        !object ||
        typeof object !== "object"
    ) {
        return "Not available";
    }

    const lines = [];

    preferredKeys.forEach(
        (key) => {
            if (
                object[key] !== undefined &&
                object[key] !== null
            ) {
                lines.push(
                    `<strong>${escapeHTML(key.replaceAll("_", " "))}:</strong>
                     ${escapeHTML(object[key])}`
                );
            }
        }
    );

    if (lines.length === 0) {
        Object.entries(object)
            .slice(0, 6)
            .forEach(
                ([key, value]) => {
                    if (
                        typeof value !== "object"
                    ) {
                        lines.push(
                            `<strong>${escapeHTML(key.replaceAll("_", " "))}:</strong>
                             ${escapeHTML(value)}`
                        );
                    }
                }
            );
    }

    return lines.join("<br>");
}


function renderPrediction(
    data
) {
    lastPrediction = data;

    hide(
        "predictionEmpty"
    );

    show(
        "predictionResult"
    );

    const location =
        `${data.city || "Unknown"}, ${data.state || "NER"}`;

    setText(
        "predictionLocation",
        location
    );

    setRiskBadge(
        data.risk_level
    );

    setText(
        "predictionRiskScore",
        `${formatNumber(
            data.risk_score,
            2
        )}%`
    );

    setText(
        "resultElevation",
        `${formatNumber(
            data.elevation_m,
            0
        )} m`
    );

    setText(
        "resultSlope",
        `${formatNumber(
            data.slope_degree,
            2
        )}°`
    );

    setText(
        "resultLandcover",
        data.landcover_class || "--"
    );

    setText(
        "resultTemperature",
        `${formatNumber(
            data.temperature_c,
            1
        )} °C`
    );

    setText(
        "resultHumidity",
        data.humidity !== undefined
            ? `${data.humidity}%`
            : "--"
    );

    setText(
        "resultPressure",
        data.pressure_hpa !== undefined
            ? `${data.pressure_hpa} hPa`
            : "--"
    );

    renderRiskDrivers(
        data.risk_drivers
    );

    $("roadResult").innerHTML =
        objectSummary(
            data.road_connectivity,
            [
                "road_status",
                "severity",
                "message",
                "recommendation"
            ]
        );

    $("emergencyResult").innerHTML =
        objectSummary(
            data.emergency_priority,
            [
                "priority_level",
                "priority_score",
                "response_action",
                "message"
            ]
        );

    setText(
        "recommendation",
        data.recommendation ||
        "Follow official advisories and field verification."
    );

    const email =
        data.automatic_email || {};

    const sms =
        data.automatic_sms ||
        email.sms ||
        {};

    function alertStatusText(
        channel,
        status
    ) {
        if (status.duplicate) {
            return {
                icon: "✅",
                state: "DUPLICATE BLOCKED",
                text: `Duplicate ${channel} alert blocked safely.`,
                className: "success"
            };
        }

        if (status.queued) {
            return {
                icon: "📶",
                state: "QUEUED",
                text:
                    status.message ||
                    `${channel} alert queued because network is unavailable.`,
                className: "queued"
            };
        }

        if (status.success) {
            return {
                icon: "✅",
                state: "SENT",
                text:
                    status.message ||
                    `${channel} alert sent successfully.`,
                className: "success"
            };
        }

        if (status.skipped) {
            return {
                icon: "ℹ️",
                state: "NOT REQUIRED",
                text:
                    status.message ||
                    `${channel} alert not required for this risk level.`,
                className: "skipped"
            };
        }

        if (status.attempted) {
            return {
                icon: "❌",
                state: "FAILED",
                text:
                    status.message ||
                    `${channel} alert failed.`,
                className: "failed"
            };
        }

        return {
            icon: "⚠️",
            state: "NOT SENT",
            text:
                status.message ||
                `${channel} alert status unavailable.`,
            className: "skipped"
        };
    }

    function renderAlertChannel(
        elementId,
        channelName,
        status
    ) {
        const box = $(elementId);

        if (!box) {
            return;
        }

        const view =
            alertStatusText(
                channelName,
                status
            );

        const riskClass =
            String(
                data.risk_level || ""
            ).toLowerCase();

        const emergencyClass =
            ["high", "critical"].includes(
                riskClass
            )
                ? riskClass
                : "";

        box.className =
            `alert-status-card ${view.className} ${emergencyClass}`.trim();

        box.innerHTML = `
            <div class="alert-channel-state">
                ${view.icon}
                ${escapeHTML(view.state)}
            </div>

            <div class="alert-title">
                ${channelName === "Email" ? "📧" : "📱"}
                ${escapeHTML(channelName)} Alert
            </div>

            <div class="alert-meta">
                ${escapeHTML(view.text)}<br>
                <strong>Location:</strong>
                ${escapeHTML(location)}<br>
                <strong>AI Risk Score:</strong>
                ${formatNumber(data.risk_score, 2)}%
            </div>
        `;
    }

    renderAlertChannel(
        "automaticEmail",
        "Email",
        email
    );

    renderAlertChannel(
        "automaticSms",
        "Twilio SMS",
        sms
    );

    setText("assistantContextTitle", `${location} • ${data.risk_level || "--"} Risk`);
    setText("assistantContextText", `AI Risk Score ${formatNumber(data.risk_score, 2)}%. The assistant will use rainfall, soil, terrain, road connectivity and emergency-priority context from this prediction.`);
}


async function submitPrediction(
    event
) {
    event.preventDefault();

    const button =
        $("predictButton");

    setMessage(
        "predictionMessage",
        "Analysing live risk...",
        "info"
    );

    setButtonLoading(
        button,
        true,
        "🚨 Calculate Landslide Risk",
        "⏳ Analysing..."
    );

    const payload = {
        state:
            $("predictionState").value,

        city:
            $("predictionCity").value.trim(),

        rainfall_24h_mm:
            numberValue("rain24"),

        rainfall_3d_mm:
            numberValue("rain3d"),

        rainfall_7d_mm:
            numberValue("rain7d"),

        soil_water_layer_1:
            numberValue("soil1"),

        soil_water_layer_2:
            numberValue("soil2")
    };

    try {
        const data =
            await apiFetch(
                "/api/predict",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body:
                        JSON.stringify(
                            payload
                        )
                }
            );

        renderPrediction(
            data
        );

        setMessage(
            "predictionMessage",
            "Risk analysis completed successfully.",
            "success"
        );
    }
    catch (error) {
        setMessage(
            "predictionMessage",
            error.message,
            "error"
        );
    }
    finally {
        setButtonLoading(
            button,
            false,
            "🚨 Calculate Landslide Risk",
            "⏳ Analysing..."
        );
    }
}


// ==========================================
// GIS
// ==========================================

function initializeMap() {
    if (
        typeof L === "undefined" ||
        gisMap
    ) {
        return;
    }

    gisMap =
        L.map(
            "gisMap"
        ).setView(
            [26.0, 93.0],
            6
        );

    L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        {
            maxZoom: 19,
            attribution:
                "&copy; OpenStreetMap contributors"
        }
    ).addTo(
        gisMap
    );

    gisLayer =
        L.layerGroup().addTo(
            gisMap
        );
}


function riskColor(
    level,
    backendColor
) {
    if (backendColor) {
        return backendColor;
    }

    const colors = {
        LOW: "#22c55e",
        MODERATE: "#f59e0b",
        HIGH: "#f97316",
        CRITICAL: "#dc2626"
    };

    return colors[
        String(level).toUpperCase()
    ] || "#64748b";
}


function renderGISPoints(
    data
) {
    initializeMap();

    gisLayer.clearLayers();

    const points =
        data.points || [];

    const bounds = [];

    points.forEach(
        (point) => {
            const latitude =
                Number(point.latitude);

            const longitude =
                Number(point.longitude);

            if (
                !Number.isFinite(latitude) ||
                !Number.isFinite(longitude)
            ) {
                return;
            }

            const color =
                riskColor(
                    point.risk_level,
                    point.marker_color
                );

            const marker =
                L.circleMarker(
                    [latitude, longitude],
                    {
                        radius: 7,
                        color: "#ffffff",
                        weight: 1.5,
                        fillColor: color,
                        fillOpacity: 0.88
                    }
                );

            marker.bindPopup(
                `
                    <strong>
                        ${escapeHTML(
                            point.state ||
                            "NER Location"
                        )}
                    </strong>
                    <br><br>

                    <strong style="color:${color}">
                        ${escapeHTML(
                            point.risk_level
                        )} RISK
                    </strong>

                    <br>
                    GIS Score:
                    ${formatNumber(
                        point.risk_score,
                        0
                    )}

                    <br>
                    Rainfall 24h:
                    ${formatNumber(
                        point.rainfall_24h_mm,
                        1
                    )} mm

                    <br>
                    Rainfall 7d:
                    ${formatNumber(
                        point.rainfall_7d_mm,
                        1
                    )} mm

                    <br>
                    Slope:
                    ${formatNumber(
                        point.slope_degree,
                        1
                    )}°

                    <br>
                    Elevation:
                    ${formatNumber(
                        point.elevation_m,
                        0
                    )} m
                `
            );

            marker.addTo(
                gisLayer
            );

            bounds.push(
                [latitude, longitude]
            );
        }
    );

    setText(
        "gisPointCount",
        data.count ?? points.length
    );

    if (bounds.length > 0) {
        gisMap.fitBounds(
            bounds,
            {
                padding: [25, 25],
                maxZoom: 9
            }
        );
    }
    else if (data.center) {
        gisMap.setView(
            [
                data.center.latitude,
                data.center.longitude
            ],
            data.zoom || 7
        );
    }

    setTimeout(
        () => gisMap.invalidateSize(),
        100
    );
}


async function loadGIS() {
    const button =
        $("loadGISButton");

    const state =
        $("gisState")?.value ||
        "All NER States";

    setButtonLoading(
        button,
        true,
        "🗺️ Load Risk Map",
        "⏳ Loading..."
    );

    setMessage(
        "gisMessage",
        "Loading dataset-backed GIS risk points...",
        "info"
    );

    try {
        const data =
            await apiFetch(
                `/api/gis/points?state=${encodeURIComponent(state)}`
            );

        renderGISPoints(
            data
        );

        setMessage(
            "gisMessage",
            `${data.count} GIS risk points loaded.`,
            "success"
        );
    }
    catch (error) {
        setMessage(
            "gisMessage",
            error.message,
            "error"
        );
    }
    finally {
        setButtonLoading(
            button,
            false,
            "🗺️ Load Risk Map",
            "⏳ Loading..."
        );
    }
}


// ==========================================
// WEATHER
// ==========================================

async function submitWeather(
    event
) {
    event.preventDefault();

    const button =
        $("weatherButton");

    const state =
        $("weatherState")?.value || "";

    const city =
        $("weatherCity").value.trim();

    setButtonLoading(
        button,
        true,
        "🌦️ Fetch Live Weather",
        "⏳ Fetching..."
    );

    setMessage(
        "weatherMessage",
        "Fetching live weather...",
        "info"
    );

    try {
        const data =
            await apiFetch(
                `/api/weather?city=${encodeURIComponent(city)}`
            );

        show(
            "weatherResult"
        );

        setText(
            "weatherTemperature",
            `${formatNumber(
                data.temperature_c,
                1
            )} °C`
        );

        setText(
            "weatherHumidity",
            data.humidity !== undefined
                ? `${data.humidity}%`
                : "--"
        );

        setText(
            "weatherPressure",
            data.pressure_hpa !== undefined
                ? `${data.pressure_hpa} hPa`
                : "--"
        );

        setText("weatherCondition", data.weather || data.condition || data.description || "--");
        setText("weatherWind", data.wind_speed !== undefined ? `${formatNumber(data.wind_speed, 1)} m/s` : (data.wind_speed_mps !== undefined ? `${formatNumber(data.wind_speed_mps,1)} m/s` : "--"));
        const visibility = data.visibility_km !== undefined ? data.visibility_km : (data.visibility !== undefined ? Number(data.visibility) / 1000 : null);
        setText("weatherVisibility", visibility !== null ? `${formatNumber(visibility,1)} km` : "--");
        const rainfall = data.rainfall_mm ?? data.rain_1h_mm ?? data.rainfall_1h_mm ?? data.rain_24h_mm;
        setText("weatherRainfall", rainfall !== undefined && rainfall !== null ? `${formatNumber(rainfall,1)} mm` : "Not reported");
        setText("weatherUpdated", data.updated_at || new Date().toLocaleTimeString("en-IN"));

        const hazardCard = $("weatherHazardCard");
        const humidity = Number(data.humidity || 0);
        const rainValue = Number(rainfall || 0);
        let hazard = "NORMAL";
        let hazardText = "Current weather does not show a strong standalone rainfall signal. Continue monitoring terrain and recent accumulated rainfall.";
        if (rainValue >= 50 || humidity >= 90) {
            hazard = "WATCH";
            hazardText = "Wet atmospheric conditions detected. Combine this with recent 3-day/7-day rainfall and slope conditions before making a landslide-risk decision.";
        }
        if (rainValue >= 100) {
            hazard = "DANGER";
            hazardText = "Very heavy reported rainfall can rapidly increase slope saturation. Check Live Risk Prediction and official local warnings immediately.";
        }
        if (hazardCard) hazardCard.className = `weather-alert-card ${hazard === "DANGER" ? "danger" : hazard === "WATCH" ? "watch" : ""}`;
        setText("weatherHazardBadge", hazard);
        setText("weatherHazardTitle", hazard === "DANGER" ? "Heavy Rainfall Hazard" : hazard === "WATCH" ? "Weather Watch" : "Weather Conditions Stable");
        setText("weatherHazardText", hazardText);

        setMessage(
            "weatherMessage",
            `Weather updated for ${data.city || city}${state ? `, ${state}` : ""}.`,
            "success"
        );
    }
    catch (error) {
        setMessage(
            "weatherMessage",
            error.message,
            "error"
        );
    }
    finally {
        setButtonLoading(
            button,
            false,
            "🌦️ Fetch Live Weather",
            "⏳ Fetching..."
        );
    }
}


// ==========================================
// DEVICE LOCATION FOR FIELD REPORT
// ==========================================

function useCurrentLocation() {
    const button = $("useMyLocationButton");
    if (!navigator.geolocation) {
        setText("locationStatus", "Geolocation is not supported by this browser.");
        return;
    }
    if (button) { button.disabled = true; button.textContent = "📍 Detecting location..."; }
    setText("locationStatus", "Waiting for browser location permission...");
    navigator.geolocation.getCurrentPosition(
        (position) => {
            $("reportLatitude").value = position.coords.latitude.toFixed(6);
            $("reportLongitude").value = position.coords.longitude.toFixed(6);
            setText("locationStatus", `Current device coordinates captured • accuracy about ${Math.round(position.coords.accuracy)} m.`);
            if (button) { button.disabled = false; button.textContent = "✅ Current Location Captured"; }
        },
        (error) => {
            const messages = {1:"Location permission was denied.",2:"Current location is unavailable.",3:"Location request timed out."};
            setText("locationStatus", messages[error.code] || "Unable to read current location.");
            if (button) { button.disabled = false; button.textContent = "📍 Use My Current Location"; }
        },
        { enableHighAccuracy:true, timeout:12000, maximumAge:30000 }
    );
}

// ==========================================
// CITIZEN REPORTS
// ==========================================

async function submitReport(
    event
) {
    event.preventDefault();

    const button =
        $("reportButton");

    const formData =
        new FormData();

    formData.append(
        "reporter_type",
        $("reporterType").value
    );

    formData.append(
        "issue_type",
        $("issueType").value
    );

    formData.append(
        "state",
        $("reportState").value
    );

    formData.append(
        "latitude",
        $("reportLatitude").value
    );

    formData.append(
        "longitude",
        $("reportLongitude").value
    );

    formData.append(
        "description",
        $("reportDescription").value
    );

    const photo =
        $("reportPhoto").files[0];

    if (photo) {
        formData.append(
            "photo",
            photo
        );
    }

    setButtonLoading(
        button,
        true,
        "📤 Submit Field Report",
        "⏳ Submitting..."
    );

    setMessage(
        "reportMessage",
        "Submitting field report...",
        "info"
    );

    try {
        await apiFetch(
            "/api/reports",
            {
                method: "POST",
                body: formData
            }
        );

        setMessage(
            "reportMessage",
            "Field report submitted successfully.",
            "success"
        );

        $("reportDescription").value = "";
        $("reportPhoto").value = "";

        await loadReports();
    }
    catch (error) {
        setMessage(
            "reportMessage",
            error.message,
            "error"
        );
    }
    finally {
        setButtonLoading(
            button,
            false,
            "📤 Submit Field Report",
            "⏳ Submitting..."
        );
    }
}


async function loadReports() {
    const container =
        $("reportsList");

    if (!container) {
        return;
    }

    try {
        const data =
            await apiFetch(
                "/api/reports"
            );

        const reports =
            data.reports || [];

        if (reports.length === 0) {
            container.innerHTML =
                `<div class="empty-state small">
                    No field reports yet.
                 </div>`;
            return;
        }

        container.innerHTML =
            reports.slice(0, 20).map(
                (report) => `
                    <div class="report-card">
                        <h4>
                            ${escapeHTML(
                                report.issue_type ||
                                report.reporter_type ||
                                "Field Report"
                            )}
                        </h4>

                        <p>
                            <strong>Reporter:</strong>
                            ${escapeHTML(
                                report.reporter_type ||
                                "--"
                            )}
                        </p>

                        <p>
                            <strong>Location:</strong>
                            ${escapeHTML(
                                report.latitude
                            )},
                            ${escapeHTML(
                                report.longitude
                            )}
                        </p>

                        <p>
                            ${escapeHTML(
                                report.description ||
                                ""
                            )}
                        </p>

                        <p>
                            ${escapeHTML(
                                report.timestamp ||
                                ""
                            )}
                        </p>
                    </div>
                `
            ).join("");
    }
    catch (error) {
        container.innerHTML =
            `<div class="message error">
                ${escapeHTML(error.message)}
             </div>`;
    }
}


// ==========================================
// GENAI
// ==========================================

function addChatMessage(
    role,
    content
) {
    const container =
        $("chatMessages");

    if (!container) {
        return;
    }

    const wrapper =
        document.createElement("div");

    wrapper.className =
        `chat-message ${role}`;

    const avatar =
        document.createElement("div");

    avatar.className =
        "chat-avatar";

    avatar.textContent =
        role === "user"
            ? "👤"
            : "🧠";

    const bubble =
        document.createElement("div");

    bubble.className =
        "chat-bubble";

    bubble.textContent =
        content;

    wrapper.appendChild(
        avatar
    );

    wrapper.appendChild(
        bubble
    );

    container.appendChild(
        wrapper
    );

    container.scrollTop =
        container.scrollHeight;
}


async function submitGenAI(
    event
) {
    event.preventDefault();

    const input =
        $("genaiInput");

    const button =
        $("genaiButton");

    const message =
        input.value.trim();

    if (!message) {
        return;
    }

    addChatMessage(
        "user",
        message
    );

    input.value = "";

    setButtonLoading(
        button,
        true,
        "Send ➜",
        "Thinking..."
    );

    setMessage(
        "genaiMessage",
        "GenAI is analysing your question...",
        "info"
    );

    const payload = {
        message: message,
        risk_context:
            lastPrediction,
        history:
            chatHistory.slice(-8)
    };

    try {
        const data =
            await apiFetch(
                "/api/genai",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body:
                        JSON.stringify(
                            payload
                        )
                }
            );

        addChatMessage(
            "assistant",
            data.answer
        );

        chatHistory.push(
            {
                role: "user",
                content: message
            },
            {
                role: "assistant",
                content: data.answer
            }
        );

        setMessage(
            "genaiMessage",
            "",
            ""
        );
    }
    catch (error) {
        addChatMessage(
            "assistant",
            `Error: ${error.message}`
        );

        setMessage(
            "genaiMessage",
            error.message,
            "error"
        );
    }
    finally {
        setButtonLoading(
            button,
            false,
            "Send ➜",
            "Thinking..."
        );
    }
}


// ==========================================
// ANALYTICS
// ==========================================

function renderStateBars(
    distribution
) {
    const container =
        $("stateBars");

    if (!container) {
        return;
    }

    const entries =
        Object.entries(
            distribution || {}
        );

    const maximum =
        Math.max(
            ...entries.map(
                ([, value]) =>
                    Number(value) || 0
            ),
            1
        );

    container.innerHTML =
        entries.map(
            ([state, value]) => {
                const width =
                    (
                        (Number(value) || 0)
                        /
                        maximum
                    ) * 100;

                return `
                    <div class="bar-row">
                        <span>${escapeHTML(state)}</span>

                        <div class="bar-track">
                            <div
                                class="bar-fill"
                                style="width:${width}%"
                            ></div>
                        </div>

                        <strong>
                            ${escapeHTML(value)}
                        </strong>
                    </div>
                `;
            }
        ).join("");
}


function renderStatsTable(
    containerId,
    statistics
) {
    const container =
        $(containerId);

    if (!container) {
        return;
    }

    const entries =
        Object.entries(
            statistics || {}
        );

    if (entries.length === 0) {
        container.innerHTML =
            `<p class="tiny">No statistics available.</p>`;
        return;
    }

    let html = `
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Mean</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
            </thead>
            <tbody>
    `;

    entries.forEach(
        ([feature, stats]) => {
            html += `
                <tr>
                    <td>${escapeHTML(feature)}</td>
                    <td>${formatNumber(stats.mean, 2)}</td>
                    <td>${formatNumber(stats.min, 2)}</td>
                    <td>${formatNumber(stats.max, 2)}</td>
                </tr>
            `;
        }
    );

    html += `
            </tbody>
        </table>
    `;

    container.innerHTML = html;
}


function renderAnalyticsPreview(
    rows
) {
    const table =
        $("analyticsTable");

    const head =
        table?.querySelector(
            "thead"
        );

    const body =
        table?.querySelector(
            "tbody"
        );

    if (
        !head ||
        !body
    ) {
        return;
    }

    if (
        !Array.isArray(rows) ||
        rows.length === 0
    ) {
        head.innerHTML = "";
        body.innerHTML =
            `<tr>
                <td>No historical rows available.</td>
             </tr>`;
        return;
    }

    const columns =
        Object.keys(
            rows[0]
        );

    head.innerHTML =
        `<tr>
            ${columns.map(
                (column) =>
                    `<th>${escapeHTML(column)}</th>`
            ).join("")}
         </tr>`;

    body.innerHTML =
        rows.map(
            (row) => `
                <tr>
                    ${columns.map(
                        (column) =>
                            `<td>
                                ${escapeHTML(
                                    row[column]
                                )}
                             </td>`
                    ).join("")}
                </tr>
            `
        ).join("");
}


async function loadAnalytics() {
    const state =
        $("analyticsState")?.value ||
        "All NER States";

    const button =
        $("loadAnalyticsButton");

    setButtonLoading(
        button,
        true,
        "📊 Load Analytics",
        "⏳ Loading..."
    );

    setMessage(
        "analyticsMessage",
        "Loading historical analytics...",
        "info"
    );

    try {
        const data =
            await apiFetch(
                `/api/analytics?state=${encodeURIComponent(state)}`
            );

        setText(
            "analyticsRecords",
            data.historical_records
        );

        setText("analyticsRegion", data.state);
        const distribution = data.state_distribution || {};
        const topEntry = Object.entries(distribution).sort((a,b) => Number(b[1]) - Number(a[1]))[0];
        setText("analyticsTopState", topEntry && Number(topEntry[1]) > 0 ? `${topEntry[0]} (${topEntry[1]})` : "--");
        const rain7 = data.rainfall_statistics?.rainfall_7d_mm;
        const slope = data.terrain_statistics?.slope_degree;
        setText("analyticsMeanRain", rain7?.mean !== undefined ? `${formatNumber(rain7.mean,1)} mm` : "--");
        setText("analyticsMeanSlope", slope?.mean !== undefined ? `${formatNumber(slope.mean,1)}°` : "--");
        setText("analyticsScope", data.state === "All NER States" ? "NER-wide" : data.state);

        renderStateBars(
            data.state_distribution
        );

        renderStatsTable(
            "rainfallStats",
            data.rainfall_statistics
        );

        renderStatsTable(
            "terrainStats",
            data.terrain_statistics
        );

        renderAnalyticsPreview(
            data.preview
        );

        setMessage(
            "analyticsMessage",
            "Historical analytics loaded.",
            "success"
        );
    }
    catch (error) {
        setMessage(
            "analyticsMessage",
            error.message,
            "error"
        );
    }
    finally {
        setButtonLoading(
            button,
            false,
            "📊 Load Analytics",
            "⏳ Loading..."
        );
    }
}


// ==========================================
// EVENT LISTENERS
// ==========================================

function initializeEvents() {
    $("predictionState")?.addEventListener(
        "change",
        updatePredictionDefaultCity
    );

    $("reportState")?.addEventListener(
        "change",
        updateReportCoordinates
    );

    $("soil1")?.addEventListener(
        "input",
        updateRangeLabels
    );

    $("soil2")?.addEventListener(
        "input",
        updateRangeLabels
    );

    $("predictionForm")?.addEventListener(
        "submit",
        submitPrediction
    );

    $("loadGISButton")?.addEventListener(
        "click",
        loadGIS
    );

    $("gisState")?.addEventListener(
        "change",
        loadGIS
    );

    $("weatherState")?.addEventListener(
        "change",
        updateWeatherCities
    );

    $("weatherForm")?.addEventListener(
        "submit",
        submitWeather
    );

    $("reportForm")?.addEventListener(
        "submit",
        submitReport
    );

    $("refreshReports")?.addEventListener("click", loadReports);

    $("useMyLocationButton")?.addEventListener("click", useCurrentLocation);

    document.querySelectorAll(".quick-prompt").forEach((button) => {
        button.addEventListener("click", () => {
            const input = $("genaiInput");
            if (input) { input.value = button.dataset.prompt || ""; input.focus(); }
        });
    });

    $("genaiForm")?.addEventListener(
        "submit",
        submitGenAI
    );

    $("loadAnalyticsButton")?.addEventListener(
        "click",
        loadAnalytics
    );

    $("analyticsState")?.addEventListener(
        "change",
        loadAnalytics
    );

    $("refreshOverview")?.addEventListener(
        "click",
        loadOverview
    );
}


// ==========================================
// START APPLICATION
// ==========================================

async function initializeApp() {
    initializeNavigation();
    initializeMap();

    updateClock();

    setInterval(
        updateClock,
        1000
    );

    try {
        await loadConfig();

        initializeEvents();

        updateRangeLabels();

        await loadOverview();

        await loadGIS();
    }
    catch (error) {
        console.error(
            "Application initialization error:",
            error
        );

        alert(
            "Backend connection failed: " +
            error.message
        );
    }
}


document.addEventListener(
    "DOMContentLoaded",
    initializeApp
);
