// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mlflow/protos/internal.proto

package org.mlflow.internal.proto;

public final class Internal {
  private Internal() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  /**
   * <pre>
   * Types of vertices represented in MLflow Run Inputs. Valid vertices are MLflow objects that can
   * have an input relationship.
   * </pre>
   *
   * Protobuf enum {@code mlflow.internal.InputVertexType}
   */
  public enum InputVertexType
      implements com.google.protobuf.ProtocolMessageEnum {
    /**
     * <code>RUN = 1;</code>
     */
    RUN(1),
    /**
     * <code>DATASET = 2;</code>
     */
    DATASET(2),
    ;

    /**
     * <code>RUN = 1;</code>
     */
    public static final int RUN_VALUE = 1;
    /**
     * <code>DATASET = 2;</code>
     */
    public static final int DATASET_VALUE = 2;


    public final int getNumber() {
      return value;
    }

    /**
     * @param value The numeric wire value of the corresponding enum entry.
     * @return The enum associated with the given numeric wire value.
     * @deprecated Use {@link #forNumber(int)} instead.
     */
    @java.lang.Deprecated
    public static InputVertexType valueOf(int value) {
      return forNumber(value);
    }

    /**
     * @param value The numeric wire value of the corresponding enum entry.
     * @return The enum associated with the given numeric wire value.
     */
    public static InputVertexType forNumber(int value) {
      switch (value) {
        case 1: return RUN;
        case 2: return DATASET;
        default: return null;
      }
    }

    public static com.google.protobuf.Internal.EnumLiteMap<InputVertexType>
        internalGetValueMap() {
      return internalValueMap;
    }
    private static final com.google.protobuf.Internal.EnumLiteMap<
        InputVertexType> internalValueMap =
          new com.google.protobuf.Internal.EnumLiteMap<InputVertexType>() {
            public InputVertexType findValueByNumber(int number) {
              return InputVertexType.forNumber(number);
            }
          };

    public final com.google.protobuf.Descriptors.EnumValueDescriptor
        getValueDescriptor() {
      return getDescriptor().getValues().get(ordinal());
    }
    public final com.google.protobuf.Descriptors.EnumDescriptor
        getDescriptorForType() {
      return getDescriptor();
    }
    public static final com.google.protobuf.Descriptors.EnumDescriptor
        getDescriptor() {
      return org.mlflow.internal.proto.Internal.getDescriptor().getEnumTypes().get(0);
    }

    private static final InputVertexType[] VALUES = values();

    public static InputVertexType valueOf(
        com.google.protobuf.Descriptors.EnumValueDescriptor desc) {
      if (desc.getType() != getDescriptor()) {
        throw new java.lang.IllegalArgumentException(
          "EnumValueDescriptor is not for this type.");
      }
      return VALUES[desc.getIndex()];
    }

    private final int value;

    private InputVertexType(int value) {
      this.value = value;
    }

    // @@protoc_insertion_point(enum_scope:mlflow.internal.InputVertexType)
  }


  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\034mlflow/protos/internal.proto\022\017mlflow.i" +
      "nternal\032#mlflow/protos/scalapb/scalapb.p" +
      "roto*\'\n\017InputVertexType\022\007\n\003RUN\020\001\022\013\n\007DATA" +
      "SET\020\002B\036\n\031org.mlflow.internal.proto\220\001\001"
    };
    descriptor = com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
          org.mlflow.scalapb_interface.Scalapb.getDescriptor(),
        });
    org.mlflow.scalapb_interface.Scalapb.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}
