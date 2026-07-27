using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

public static class BuildQuestApk
{
    private const string OutputPath = "Builds/DobotVrTeleop.apk";

    [MenuItem("Build/Build Quest APK")]
    public static void Build()
    {
        string[] scenes = EditorBuildSettings.scenes
            .Where(scene => scene.enabled)
            .Select(scene => scene.path)
            .ToArray();

        if (scenes.Length == 0)
        {
            throw new InvalidOperationException("No enabled scenes are configured for the build.");
        }

        string absoluteOutputPath = Path.GetFullPath(
            Path.Combine(Application.dataPath, "..", OutputPath));
        Directory.CreateDirectory(Path.GetDirectoryName(absoluteOutputPath));

        EditorUserBuildSettings.buildAppBundle = false;
        BuildPlayerOptions options = new BuildPlayerOptions
        {
            scenes = scenes,
            locationPathName = absoluteOutputPath,
            target = BuildTarget.Android,
            options = BuildOptions.None,
        };

        BuildReport report = BuildPipeline.BuildPlayer(options);
        if (report.summary.result != BuildResult.Succeeded)
        {
            throw new InvalidOperationException(
                $"Quest APK build failed: {report.summary.result}, " +
                $"errors={report.summary.totalErrors}, warnings={report.summary.totalWarnings}");
        }

        Debug.Log($"Quest APK built successfully: {absoluteOutputPath}");
    }
}
