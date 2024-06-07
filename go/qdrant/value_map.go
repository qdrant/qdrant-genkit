package qdrant

// This file contains methods to convert a generic map[string]any to map[string]*grpc.Value(Qdrant payload type).
// This is a custom implementatation based on "google.golang.org/protobuf/types/known/structpb".
// It extends the original implementation to support IntegerValue and DoubleValue instead of a single NumberValue.
//
// USAGE:
//
// jsonMap := map[string]any{
// 	"some_null":    nil,
// 	"some_bool":    true,
// 	"some_int":     42,
// 	"some_float":   3.14,
// 	"some_string":  "hello",
// 	"some_bytes":   []byte("world"),
// 	"some_nested":  map[string]any{"key": "value"},
// 	"some_list":    []any{"foo", 32},
// }
//
// valueMap := newValueMap(jsonMap)

import (
	"encoding/base64"
	"fmt"
	"unicode/utf8"

	grpc "github.com/qdrant/go-client/qdrant"
)

// Converts a map of string to any to a map of string to *grpc.Value
//
//	╔════════════════════════╤════════════════════════════════════════════╗
//	║ Go type                │ Conversion                                 ║
//	╠════════════════════════╪════════════════════════════════════════════╣
//	║ nil                    │ stored as NullValue                        ║
//	║ bool                   │ stored as BoolValue                        ║
//	║ int, int32, int64      │ stored as IntegerValue                     ║
//	║ uint, uint32, uint64   │ stored as IntegerValue                     ║
//	║ float32, float64       │ stored as DoubleValue                      ║
//	║ string                 │ stored as StringValue; must be valid UTF-8 ║
//	║ []byte                 │ stored as StringValue; base64-encoded      ║
//	║ map[string]any │ stored as StructValue                      ║
//	║ []any          │ stored as ListValue                        ║
//	╚════════════════════════╧════════════════════════════════════════════╝

func newValueMap(inputMap map[string]any) map[string]*grpc.Value {
	valueMap := make(map[string]*grpc.Value)
	for key, val := range inputMap {
		value, err := newValue(val)
		if err != nil {
			panic(err)
		}
		valueMap[key] = value
	}
	return valueMap
}

// newValue constructs a *grpc.Value from a general-purpose Go interface.
func newValue(v any) (*grpc.Value, error) {
	switch v := v.(type) {
	case nil:
		return newNullValue(), nil
	case bool:
		return newBoolValue(v), nil
	case int:
		return newIntegerValue(int64(v)), nil
	case int32:
		return newIntegerValue(int64(v)), nil
	case int64:
		return newIntegerValue(int64(v)), nil
	case uint:
		return newIntegerValue(int64(v)), nil
	case uint32:
		return newIntegerValue(int64(v)), nil
	case uint64:
		return newIntegerValue(int64(v)), nil
	case float32:
		return newDoubleValue(float64(v)), nil
	case float64:
		return newDoubleValue(float64(v)), nil
	case string:
		if !utf8.ValidString(v) {
			return nil, fmt.Errorf("invalid UTF-8 in string: %q", v)
		}
		return newStringValue(v), nil
	case []byte:
		s := base64.StdEncoding.EncodeToString(v)
		return newStringValue(s), nil
	case map[string]any:
		v2, err := newStruct(v)
		if err != nil {
			return nil, err
		}
		return newStructValue(v2), nil
	case []any:
		v2, err := newList(v)
		if err != nil {
			return nil, err
		}
		return newListValue(v2), nil
	default:
		return nil, fmt.Errorf("invalid type: %T", v)
	}
}

// newNullValue constructs a new null Value.
func newNullValue() *grpc.Value {
	return &grpc.Value{Kind: &grpc.Value_NullValue{NullValue: grpc.NullValue_NULL_VALUE}}
}

// newBoolValue constructs a new boolean Value.
func newBoolValue(v bool) *grpc.Value {
	return &grpc.Value{Kind: &grpc.Value_BoolValue{BoolValue: v}}
}

// newInteger constructs a new number Value.
func newIntegerValue(v int64) *grpc.Value {
	return &grpc.Value{Kind: &grpc.Value_IntegerValue{IntegerValue: v}}
}

// newNumberValue constructs a new number Value.
func newDoubleValue(v float64) *grpc.Value {
	return &grpc.Value{Kind: &grpc.Value_DoubleValue{DoubleValue: v}}
}

// newStringValue constructs a new string Value.
func newStringValue(v string) *grpc.Value {
	return &grpc.Value{Kind: &grpc.Value_StringValue{StringValue: v}}
}

// newStructValue constructs a new struct Value.
func newStructValue(v *grpc.Struct) *grpc.Value {
	return &grpc.Value{Kind: &grpc.Value_StructValue{StructValue: v}}
}

// newListValue constructs a new list Value.
func newListValue(v *grpc.ListValue) *grpc.Value {
	return &grpc.Value{Kind: &grpc.Value_ListValue{ListValue: v}}
}

// newList constructs a ListValue from a general-purpose Go slice.
// The slice elements are converted using newValue.
func newList(v []any) (*grpc.ListValue, error) {
	x := &grpc.ListValue{Values: make([]*grpc.Value, len(v))}
	for i, v := range v {
		var err error
		x.Values[i], err = newValue(v)
		if err != nil {
			return nil, err
		}
	}
	return x, nil
}

// newStruct constructs a Struct from a general-purpose Go map.
// The map keys must be valid UTF-8.
// The map values are converted using newValue.
func newStruct(v map[string]any) (*grpc.Struct, error) {
	x := &grpc.Struct{Fields: make(map[string]*grpc.Value, len(v))}
	for k, v := range v {
		if !utf8.ValidString(k) {
			return nil, fmt.Errorf("invalid UTF-8 in string: %q", k)
		}
		var err error
		x.Fields[k], err = newValue(v)
		if err != nil {
			return nil, err
		}
	}
	return x, nil
}
