using System;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using UnityEngine.XR;

public class QuestVrUdpSender : MonoBehaviour
{
    [Header("Network")]
    [Tooltip("Used only when automatic discovery is disabled or manual fallback is enabled.")]
    public string targetIp = "192.168.89.66";
    public int targetPort = 5005;
    public float sendRateHz = 50.0f;
    public bool autoDiscoverHost = true;
    public int discoveryPort = 5006;
    public float discoveryIntervalSeconds = 1.5f;
    public float discoveryTimeoutSeconds = 10.0f;
    public bool useManualTargetFallback = false;

    [Header("Input")]
    public XRNode controllerNode = XRNode.RightHand;

    [Header("Headset-free operation")]
    [Tooltip("Keep the Quest display and app awake while the teleoperation app is open.")]
    public bool keepHeadsetAwake = true;

    private UdpClient udp;
    private UdpClient discoveryUdp;
    private IPEndPoint endpoint;
    private float nextSendTime;
    private float nextDiscoveryTime;
    private float discoveryDeadline;
    private bool discoveryClosed;
    private bool hasDiscoveredEndpoint;
    private readonly object discoveryLock = new object();
    private string pendingDiscoveryIp;
    private int pendingDiscoveryPort;

    private const string DiscoveryRequest = "DOBOT_CRTRAINER_DISCOVER_V1";

    [Serializable]
    private class DiscoveryResponse
    {
        public string type;
        public int version;
        public int port;
    }

    void Awake()
    {
        Application.runInBackground = true;
        if (!keepHeadsetAwake)
        {
            return;
        }

        Screen.sleepTimeout = SleepTimeout.NeverSleep;
        KeepAndroidWindowAwake();
    }

    void Start()
    {
        udp = new UdpClient();
        if (autoDiscoverHost)
        {
            discoveryUdp = new UdpClient(0);
            discoveryUdp.EnableBroadcast = true;
            discoveryDeadline = Time.unscaledTime + Mathf.Max(discoveryTimeoutSeconds, 0.0f);
            BeginDiscoveryReceive();
        }
        else
        {
            SetEndpoint(targetIp, targetPort, false);
        }
    }

    void OnDestroy()
    {
        discoveryClosed = true;
        udp?.Close();
        discoveryUdp?.Close();
    }

    private void KeepAndroidWindowAwake()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        try
        {
            using AndroidJavaClass unityPlayer = new AndroidJavaClass("com.unity3d.player.UnityPlayer");
            AndroidJavaObject activity = unityPlayer.GetStatic<AndroidJavaObject>("currentActivity");
            activity.Call("runOnUiThread", new AndroidJavaRunnable(() =>
            {
                AndroidJavaObject window = activity.Call<AndroidJavaObject>("getWindow");
                window.Call("addFlags", 128); // WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON
            }));
        }
        catch (Exception exception)
        {
            Debug.LogWarning($"Could not keep the Android window awake: {exception.Message}");
        }
#endif
    }

    void Update()
    {
        ApplyPendingDiscovery();
        if (autoDiscoverHost)
        {
            if (Time.unscaledTime >= nextDiscoveryTime)
            {
                SendDiscoveryRequest();
                nextDiscoveryTime = Time.unscaledTime + Mathf.Max(discoveryIntervalSeconds, 0.2f);
            }

            if (!hasDiscoveredEndpoint && useManualTargetFallback &&
                Time.unscaledTime >= discoveryDeadline && !string.IsNullOrWhiteSpace(targetIp))
            {
                SetEndpoint(targetIp, targetPort, true);
            }
        }

        if (endpoint == null)
        {
            return;
        }

        float now = Time.unscaledTime;
        float sendInterval = 1.0f / Mathf.Max(sendRateHz, 1.0f);
        if (now < nextSendTime)
        {
            return;
        }

        // Advance the periodic schedule instead of restarting it from now.
        // Restarting here quantizes 50 Hz to about 36 Hz on a 72 Hz headset.
        nextSendTime += sendInterval;
        if (nextSendTime <= now - sendInterval)
        {
            nextSendTime = now + sendInterval;
        }

        InputDevice device = InputDevices.GetDeviceAtXRNode(controllerNode);
        bool hasPosition = device.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 pos);
        bool hasRotation = device.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion rot);
        device.TryGetFeatureValue(CommonUsages.grip, out float grip);
        device.TryGetFeatureValue(CommonUsages.trigger, out float trigger);
        device.TryGetFeatureValue(CommonUsages.primaryButton, out bool buttonA);
        device.TryGetFeatureValue(CommonUsages.secondaryButton, out bool buttonB);
        device.TryGetFeatureValue(CommonUsages.primary2DAxis, out Vector2 stick);
        bool trackingValid = device.isValid && hasPosition && hasRotation;

        string json = FormatPacket(
            trackingValid,
            pos,
            rot,
            grip,
            trigger,
            buttonA,
            buttonB,
            stick
        );
        byte[] bytes = Encoding.UTF8.GetBytes(json);
        udp.Send(bytes, bytes.Length, endpoint);
    }

    private void BeginDiscoveryReceive()
    {
        if (discoveryUdp == null || discoveryClosed)
        {
            return;
        }

        try
        {
            discoveryUdp.BeginReceive(OnDiscoveryReceived, null);
        }
        catch (ObjectDisposedException)
        {
        }
    }

    private void OnDiscoveryReceived(System.IAsyncResult result)
    {
        if (discoveryUdp == null || discoveryClosed)
        {
            return;
        }

        try
        {
            IPEndPoint remote = new IPEndPoint(IPAddress.Any, 0);
            byte[] payload = discoveryUdp.EndReceive(result, ref remote);
            DiscoveryResponse response = JsonUtility.FromJson<DiscoveryResponse>(
                Encoding.UTF8.GetString(payload));
            if (response != null && response.type == "dobot_crtrainer_host" &&
                response.version == 1 && response.port > 0 && response.port <= 65535)
            {
                lock (discoveryLock)
                {
                    pendingDiscoveryIp = remote.Address.ToString();
                    pendingDiscoveryPort = response.port;
                }
            }
        }
        catch (ObjectDisposedException)
        {
            return;
        }
        catch (Exception exception)
        {
            Debug.LogWarning($"Quest host discovery failed: {exception.Message}");
        }
        finally
        {
            BeginDiscoveryReceive();
        }
    }

    private void SendDiscoveryRequest()
    {
        if (discoveryUdp == null || discoveryClosed)
        {
            return;
        }

        byte[] payload = Encoding.UTF8.GetBytes(DiscoveryRequest);
        try
        {
            discoveryUdp.Send(
                payload,
                payload.Length,
                new IPEndPoint(IPAddress.Broadcast, discoveryPort));
        }
        catch (Exception exception)
        {
            Debug.LogWarning($"Quest host discovery send failed: {exception.Message}");
        }
    }

    private void ApplyPendingDiscovery()
    {
        string discoveredIp = null;
        int discoveredPort = 0;
        lock (discoveryLock)
        {
            if (!string.IsNullOrEmpty(pendingDiscoveryIp))
            {
                discoveredIp = pendingDiscoveryIp;
                discoveredPort = pendingDiscoveryPort;
                pendingDiscoveryIp = null;
            }
        }

        if (!string.IsNullOrEmpty(discoveredIp))
        {
            SetEndpoint(discoveredIp, discoveredPort, true);
            hasDiscoveredEndpoint = true;
            Debug.Log($"Quest discovered Dobot host {discoveredIp}:{discoveredPort}");
        }
    }

    private void SetEndpoint(string ip, int port, bool discovered)
    {
        try
        {
            endpoint = new IPEndPoint(IPAddress.Parse(ip), port);
            if (!discovered)
            {
                Debug.Log($"Quest using manual Dobot host {ip}:{port}");
            }
        }
        catch (Exception exception)
        {
            endpoint = null;
            Debug.LogWarning($"Invalid Dobot host {ip}:{port}: {exception.Message}");
        }
    }

    private string FormatPacket(
        bool trackingValid,
        Vector3 pos,
        Quaternion rot,
        float grip,
        float trigger,
        bool buttonA,
        bool buttonB,
        Vector2 stick)
    {
        CultureInfo c = CultureInfo.InvariantCulture;
        double timestamp = Time.realtimeSinceStartupAsDouble;
        return string.Format(
            c,
            "{{\"timestamp\":{0:F6},\"tracking_valid\":{1},\"right_pos\":[{2:F6},{3:F6},{4:F6}],\"right_rot\":[{5:F6},{6:F6},{7:F6},{8:F6}],\"right_grip\":{9:F4},\"right_trigger\":{10:F4},\"button_a\":{11},\"button_b\":{12},\"right_thumbstick\":[{13:F4},{14:F4}]}}",
            timestamp,
            trackingValid ? "true" : "false",
            pos.x, pos.y, pos.z,
            rot.x, rot.y, rot.z, rot.w,
            grip,
            trigger,
            buttonA ? "true" : "false",
            buttonB ? "true" : "false",
            stick.x, stick.y
        );
    }
}
