; Inno Setup script. Built by .github/workflows/installer.yml with
;   ISCC /DAppName=... /DAppVersion=... /DPublisher=... /DAppGuid=... /DCliExe=... /DGuiExe=... /DSourceDir=... /DOutDir=... /DOutName=... /DExpires=...
; Per user, no administrator rights: installs to %LOCALAPPDATA%\Programs\<name>, adds a Start-menu shortcut and an
; uninstaller. No service, no driver, no autostart entry, nothing in the machine-wide registry.

#ifndef AppName
  #define AppName "StreamSafe"
#endif
#ifndef AppVersion
  #define AppVersion "0.0.0.0"
#endif
#ifndef Publisher
  #define Publisher "StreamSafe"
#endif
#ifndef AppGuid
  #define AppGuid "6F0C4B0E-5B0A-4E0B-9D53-5E1B7E2C8A41"
#endif
; Inno writes a literal opening brace as "{{"
#define AppIdValue "{{" + AppGuid + "}"
#ifndef CliExe
  #define CliExe "StreamSafe"
#endif
#ifndef GuiExe
  #define GuiExe "StreamSafeApp"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\StreamSafe"
#endif
#ifndef OutDir
  #define OutDir "..\dist-installer"
#endif
#ifndef OutName
  #define OutName "StreamSafe-Beta-Setup"
#endif
#ifndef Expires
  #define Expires ""
#endif

[Setup]
AppId={#AppIdValue}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion} (beta)
AppPublisher={#Publisher}
VersionInfoVersion={#AppVersion}
VersionInfoCompany={#Publisher}
VersionInfoDescription={#AppName} setup
VersionInfoProductName={#AppName}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutDir}
OutputBaseFilename={#OutName}
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#GuiExe}.exe
UninstallDisplayName={#AppName}
AppMutex={#AppName}-window
CloseApplications=no
InfoBeforeFile=licenses\NOTICE.txt
ShowLanguageDialog=auto
LanguageDetectionMethod=uilanguage

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "ar"; MessagesFile: "compiler:Languages\Arabic.isl"

[CustomMessages]
en.DeleteData=Also delete the downloaded models, engines, reports and settings of {#AppName} (several GB)?
ar.DeleteData=هل تريد أيضًا حذف النماذج المنزّلة وملفات التجهيز والتقارير والإعدادات الخاصة بـ {#AppName} (عدة جيجابايت)؟
en.OpenApp=Open {#AppName}
ar.OpenApp=افتح {#AppName}

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\obs\streamsafe_guard.lua"; DestDir: "{app}\obs"; Flags: ignoreversion
; StreamSafe's own logo on the covers (paths.bundled_brand_dir)
Source: "..\marketing\logo.png"; DestDir: "{app}\brand"; Flags: ignoreversion
Source: "..\marketing\logo_rgba.png"; DestDir: "{app}\brand"; Flags: ignoreversion
Source: "licenses\*"; DestDir: "{app}\licenses"; Flags: ignoreversion
Source: "..\NOTICE.md"; DestDir: "{app}\licenses"; Flags: ignoreversion
; the risk heads (ours; not downloaded). A test build carries the PLACEHOLDER marker among them.
Source: "..\build\heads\*"; DestDir: "{app}\heads"; Flags: ignoreversion
; LGPL FFmpeg, pinned by sha256 (installer\ffmpeg.lock.json)
Source: "..\build\ffmpeg\*"; DestDir: "{app}\ffmpeg"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#GuiExe}.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#GuiExe}.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Run]
Filename: "{app}\{#GuiExe}.exe"; Description: "{cm:OpenApp}"; Flags: nowait postinstall skipifsilent

[Code]
// Uninstall: the program folder always goes. The user's own data (models, settings, reports; one folder under
// %LOCALAPPDATA%) goes too when the user says yes. A SILENT uninstall removes it as well, unless /KEEPDATA is given,
// so that "silent install, silent uninstall" leaves nothing behind.
function WantsDataKept(): Boolean;
var
  i: Integer;
begin
  Result := False;
  for i := 1 to ParamCount do
    if CompareText(ParamStr(i), '/KEEPDATA') = 0 then Result := True;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
  Delete: Boolean;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\{#AppName}');
    if not DirExists(DataDir) then Exit;
    if UninstallSilent then
      Delete := not WantsDataKept()
    else
      Delete := MsgBox(CustomMessage('DeleteData'), mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES;
    if Delete then DelTree(DataDir, True, True, True);
  end;
end;
