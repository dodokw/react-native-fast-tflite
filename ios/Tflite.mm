#import "Tflite.h"
#import "../cpp/TensorflowPlugin.h"
#import <React-callinvoker/ReactCommon/CallInvoker.h>
#import <React/RCTBridge+Private.h>
#import <ReactCommon/RCTTurboModuleWithJSIBindings.h>
#import <jsi/jsi.h>
#import <string>

using namespace facebook;

// ---------------------------------------------------------------
// In bridgeless mode the TurboModule infrastructure calls
// -installJSIBindingsWithRuntime:callInvoker: automatically,
// giving us the jsi::Runtime and CallInvoker without needing
// RCTBridge at all.
//
// The JS-side `install()` method becomes a no-op that just
// returns whether the bindings were already installed.
// ---------------------------------------------------------------

@interface Tflite () <RCTTurboModuleWithJSIBindings>
@end

@implementation Tflite {
  BOOL _installed;
}

RCT_EXPORT_MODULE(Tflite)

#pragma mark - RCTTurboModuleWithJSIBindings

- (void)installJSIBindingsWithRuntime:(jsi::Runtime &)runtime
                          callInvoker:(const std::shared_ptr<react::CallInvoker> &)callInvoker {
  NSLog(@"[TFLite] installJSIBindingsWithRuntime:callInvoker: called");

  auto fetchByteDataFromUrl = [](std::string url) {
    NSString* string = [NSString stringWithUTF8String:url.c_str()];
    NSLog(@"Fetching %@...", string);
    NSURL* nsURL = [NSURL URLWithString:string];
    NSData* contents = [NSData dataWithContentsOfURL:nsURL];

    void* data = malloc(contents.length * sizeof(uint8_t));
    memcpy(data, contents.bytes, contents.length);
    return Buffer{.data = data, .size = contents.length};
  };

  try {
    TensorflowPlugin::installToRuntime(runtime, callInvoker, fetchByteDataFromUrl);
    _installed = YES;
    NSLog(@"[TFLite] Successfully installed JSI Bindings via RCTTurboModuleWithJSIBindings!");
  } catch (std::exception& exc) {
    NSLog(@"[TFLite] Failed to install TensorFlow Lite plugin! %s", exc.what());
    _installed = NO;
  }
}

#pragma mark - TurboModule install() method (called from JS)

- (NSNumber *)install {
  // In bridgeless mode, installJSIBindingsWithRuntime:callInvoker:
  // is called automatically before any JS runs, so _installed should
  // already be YES by the time JS calls install().
  if (_installed) {
    NSLog(@"[TFLite] install() called from JS — bindings already installed.");
    return @(true);
  }

  // Fallback for bridge mode: try the old approach
  NSLog(@"[TFLite] install() called from JS — attempting bridge-based installation...");

  @try {
    RCTBridge *bridge = [RCTBridge currentBridge];
    RCTCxxBridge *cxxBridge = (RCTCxxBridge *)bridge;
    if (!cxxBridge.runtime) {
      NSLog(@"[TFLite] ERROR: Could not obtain JSI runtime via bridge.");
      return @(false);
    }
    jsi::Runtime& runtime = *(jsi::Runtime *)cxxBridge.runtime;

    auto fetchByteDataFromUrl = [](std::string url) {
      NSString* string = [NSString stringWithUTF8String:url.c_str()];
      NSLog(@"Fetching %@...", string);
      NSURL* nsURL = [NSURL URLWithString:string];
      NSData* contents = [NSData dataWithContentsOfURL:nsURL];

      void* data = malloc(contents.length * sizeof(uint8_t));
      memcpy(data, contents.bytes, contents.length);
      return Buffer{.data = data, .size = contents.length};
    };

    TensorflowPlugin::installToRuntime(runtime, [bridge jsCallInvoker], fetchByteDataFromUrl);
    _installed = YES;
    NSLog(@"[TFLite] Successfully installed via bridge fallback!");
    return @(true);
  } @catch (NSException *exception) {
    NSLog(@"[TFLite] Bridge fallback also failed: %@", exception.reason);
    return @(false);
  }
}

// Don't compile this code when we build for the old architecture.
#ifdef RCT_NEW_ARCH_ENABLED
- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams&)params {
  return std::make_shared<facebook::react::NativeRNTfliteSpecJSI>(params);
}
#endif

@end
