// Code generated by protoc-gen-go-grpc. DO NOT EDIT.

package client

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
const _ = grpc.SupportPackageIsVersion7

// DatabricksMlflowArtifactsServiceClient is the client API for DatabricksMlflowArtifactsService service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type DatabricksMlflowArtifactsServiceClient interface {
	// Fetch credentials to read from the specified MLflow artifact location
	//
	// Note: Even if no artifacts exist at the specified artifact location, this API will
	// still provide read credentials as long as the format of the location is valid.
	// Callers must subsequently check for the existence of the artifacts using the appropriate
	// cloud storage APIs (as determined by the `ArtifactCredentialType` property of the response)
	GetCredentialsForRead(ctx context.Context, in *GetCredentialsForRead, opts ...grpc.CallOption) (*GetCredentialsForRead_Response, error)
	// Fetch credentials to write to the specified MLflow artifact location
	GetCredentialsForWrite(ctx context.Context, in *GetCredentialsForWrite, opts ...grpc.CallOption) (*GetCredentialsForWrite_Response, error)
}

type databricksMlflowArtifactsServiceClient struct {
	cc grpc.ClientConnInterface
}

func NewDatabricksMlflowArtifactsServiceClient(cc grpc.ClientConnInterface) DatabricksMlflowArtifactsServiceClient {
	return &databricksMlflowArtifactsServiceClient{cc}
}

func (c *databricksMlflowArtifactsServiceClient) GetCredentialsForRead(ctx context.Context, in *GetCredentialsForRead, opts ...grpc.CallOption) (*GetCredentialsForRead_Response, error) {
	out := new(GetCredentialsForRead_Response)
	err := c.cc.Invoke(ctx, "/mlflow.DatabricksMlflowArtifactsService/getCredentialsForRead", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *databricksMlflowArtifactsServiceClient) GetCredentialsForWrite(ctx context.Context, in *GetCredentialsForWrite, opts ...grpc.CallOption) (*GetCredentialsForWrite_Response, error) {
	out := new(GetCredentialsForWrite_Response)
	err := c.cc.Invoke(ctx, "/mlflow.DatabricksMlflowArtifactsService/getCredentialsForWrite", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// DatabricksMlflowArtifactsServiceServer is the server API for DatabricksMlflowArtifactsService service.
// All implementations must embed UnimplementedDatabricksMlflowArtifactsServiceServer
// for forward compatibility
type DatabricksMlflowArtifactsServiceServer interface {
	// Fetch credentials to read from the specified MLflow artifact location
	//
	// Note: Even if no artifacts exist at the specified artifact location, this API will
	// still provide read credentials as long as the format of the location is valid.
	// Callers must subsequently check for the existence of the artifacts using the appropriate
	// cloud storage APIs (as determined by the `ArtifactCredentialType` property of the response)
	GetCredentialsForRead(context.Context, *GetCredentialsForRead) (*GetCredentialsForRead_Response, error)
	// Fetch credentials to write to the specified MLflow artifact location
	GetCredentialsForWrite(context.Context, *GetCredentialsForWrite) (*GetCredentialsForWrite_Response, error)
	mustEmbedUnimplementedDatabricksMlflowArtifactsServiceServer()
}

// UnimplementedDatabricksMlflowArtifactsServiceServer must be embedded to have forward compatible implementations.
type UnimplementedDatabricksMlflowArtifactsServiceServer struct {
}

func (UnimplementedDatabricksMlflowArtifactsServiceServer) GetCredentialsForRead(context.Context, *GetCredentialsForRead) (*GetCredentialsForRead_Response, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetCredentialsForRead not implemented")
}
func (UnimplementedDatabricksMlflowArtifactsServiceServer) GetCredentialsForWrite(context.Context, *GetCredentialsForWrite) (*GetCredentialsForWrite_Response, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetCredentialsForWrite not implemented")
}
func (UnimplementedDatabricksMlflowArtifactsServiceServer) mustEmbedUnimplementedDatabricksMlflowArtifactsServiceServer() {
}

// UnsafeDatabricksMlflowArtifactsServiceServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to DatabricksMlflowArtifactsServiceServer will
// result in compilation errors.
type UnsafeDatabricksMlflowArtifactsServiceServer interface {
	mustEmbedUnimplementedDatabricksMlflowArtifactsServiceServer()
}

func RegisterDatabricksMlflowArtifactsServiceServer(s *grpc.Server, srv DatabricksMlflowArtifactsServiceServer) {
	s.RegisterService(&_DatabricksMlflowArtifactsService_serviceDesc, srv)
}

func _DatabricksMlflowArtifactsService_GetCredentialsForRead_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(GetCredentialsForRead)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(DatabricksMlflowArtifactsServiceServer).GetCredentialsForRead(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/mlflow.DatabricksMlflowArtifactsService/GetCredentialsForRead",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(DatabricksMlflowArtifactsServiceServer).GetCredentialsForRead(ctx, req.(*GetCredentialsForRead))
	}
	return interceptor(ctx, in, info, handler)
}

func _DatabricksMlflowArtifactsService_GetCredentialsForWrite_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(GetCredentialsForWrite)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(DatabricksMlflowArtifactsServiceServer).GetCredentialsForWrite(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/mlflow.DatabricksMlflowArtifactsService/GetCredentialsForWrite",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(DatabricksMlflowArtifactsServiceServer).GetCredentialsForWrite(ctx, req.(*GetCredentialsForWrite))
	}
	return interceptor(ctx, in, info, handler)
}

var _DatabricksMlflowArtifactsService_serviceDesc = grpc.ServiceDesc{
	ServiceName: "mlflow.DatabricksMlflowArtifactsService",
	HandlerType: (*DatabricksMlflowArtifactsServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "getCredentialsForRead",
			Handler:    _DatabricksMlflowArtifactsService_GetCredentialsForRead_Handler,
		},
		{
			MethodName: "getCredentialsForWrite",
			Handler:    _DatabricksMlflowArtifactsService_GetCredentialsForWrite_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "databricks_artifacts.proto",
}
